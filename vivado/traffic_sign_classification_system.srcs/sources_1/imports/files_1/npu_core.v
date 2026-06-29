// =====================================================================
//  npu_core.v   교통표지판 CNN 추론 엔진 (고정소수점 Q7/Q14)
//  구조: 28x28x1 -Conv5- 28x28x5 -Pool- 14x14x5 -Conv6- 14x14x6
//        -Pool- 7x7x6 -Flatten(294)- Dense16 - Dense10(logit)
//  시퀀셜 MAC 1개를 재사용 -> 작은 회로. ~약 9만 클럭/추론.
// =====================================================================
module npu_core (
    input  wire        clk,
    input  wire        rst,         // 동기 리셋 (active-high)
    input  wire        start,       // 1클럭 펄스 -> 추론 시작
    // 입력 이미지 쓰기 포트 (UART 수신부가 채움)
    input  wire        img_we,
    input  wire [9:0]  img_waddr,   // 0~783
    input  wire [7:0]  img_wdata,   // Q7 픽셀 (0~127)
    // 결과
    output reg         done,        // 추론 완료시 1 (level, start로 clear)
    output reg  [3:0]  pred_class,  // 0~9
    // 결과 logit 읽기 포트 (UART 송신부가 사용)
    input  wire [3:0]  rd_idx,      // 0~9
    output wire signed [31:0] rd_logit
);
    // ---------------- 메모리 ----------------
    // 입력/중간 활성화 (async read 분산 RAM)
    reg  [7:0]         img_mem [0:783];    // 28*28
    reg signed [15:0]  c1_mem  [0:3919];   // 28*28*5
    reg signed [15:0]  p1_mem  [0:979];    // 14*14*5
    reg signed [15:0]  c2_mem  [0:1175];   // 14*14*6
    reg signed [15:0]  p2_mem  [0:293];    // 7*7*6  (flatten, HWC)
    reg signed [15:0]  d1_mem  [0:15];     // 16
    reg signed [31:0]  logit_mem [0:9];    // 10  (Q14)
    // 가중치/바이어스 ROM (model_to_text.py가 만든 .mem)
    reg signed [7:0]   c1w [0:44];   reg signed [31:0] c1b [0:4];
    reg signed [7:0]   c2w [0:269];  reg signed [31:0] c2b [0:5];
    reg signed [7:0]   d1w [0:4703]; reg signed [31:0] d1b [0:15];
    reg signed [7:0]   d2w [0:159];  reg signed [31:0] d2b [0:9];

    initial begin
        $readmemh("conv1_w.mem", c1w); $readmemh("conv1_b.mem", c1b);
        $readmemh("conv2_w.mem", c2w); $readmemh("conv2_b.mem", c2b);
        $readmemh("dense1_w.mem", d1w);$readmemh("dense1_b.mem", d1b);
        $readmemh("dense2_w.mem", d2w);$readmemh("dense2_b.mem", d2b);
    end

    // 입력 이미지 쓰기
    always @(posedge clk)
        if (img_we) img_mem[img_waddr] <= img_wdata;

    assign rd_logit = logit_mem[rd_idx];

    // ---------------- FSM 상태 ----------------
    localparam S_IDLE=0,
               C1_INIT=1, C1_MAC=2, C1_WB=3, P1=4,
               C2_INIT=5, C2_MAC=6, C2_WB=7, P2=8,
               D1_INIT=9, D1_MAC=10,D1_WB=11,
               D2_INIT=12,D2_MAC=13,D2_WB=14,
               ARG=15, S_DONE=16;
    reg [4:0] state;

    // 카운터 (conv: oh,ow,oc,kh,kw,ic / dense: j,i / pool: oh,ow,cc)
    reg [4:0] oh, ow;     // 0~27
    reg [2:0] oc;         // 0~5
    reg [1:0] kh, kw;     // 0~2
    reg [2:0] ic;         // 0~4
    reg [12:0] jj, ii;    // dense index (최대 294)
    reg signed [39:0] acc;

    // 현재 conv 레이어 파라미터 (C1: layer=0, C2: layer=1)
    // H,W는 conv 출력=입력 크기, Cin/Cout
    wire        c_is2 = (state==C2_INIT)||(state==C2_MAC)||(state==C2_WB);
    wire [4:0]  cH    = c_is2 ? 5'd14 : 5'd28;
    wire [4:0]  cW    = c_is2 ? 5'd14 : 5'd28;
    wire [2:0]  cCin  = c_is2 ? 3'd5  : 3'd1;
    wire [2:0]  cCout = c_is2 ? 3'd6  : 3'd5;

    // --- conv 입력 픽셀 읽기 (same padding: 범위 밖이면 0) ---
    wire signed [5:0] ih = $signed({1'b0,oh}) + $signed({4'b0,kh}) - 6'sd1;
    wire signed [5:0] iw = $signed({1'b0,ow}) + $signed({4'b0,kw}) - 6'sd1;
    wire in_range = (ih>=0)&&(ih<$signed({1'b0,cH}))&&(iw>=0)&&(iw<$signed({1'b0,cW}));
    wire [10:0] in_addr = (ih*cW + iw)*cCin + ic;   // 평탄화 주소
    reg  signed [15:0] in_val;
    always @(*) begin
        if (!in_range)      in_val = 16'sd0;
        else if (!c_is2)    in_val = {8'd0, img_mem[in_addr[9:0]]}; // C1: 8bit 입력
        else                in_val = p1_mem[in_addr[9:0]];          // C2: 16bit 입력
    end
    // --- conv 가중치 주소: ((oc*3+kh)*3+kw)*Cin+ic ---
    wire [12:0] cw_addr = ((oc*3+kh)*3+kw)*cCin + ic;
    wire signed [7:0] cw_val = c_is2 ? c2w[cw_addr] : c1w[cw_addr[8:0]];
    wire signed [31:0] cb_val = c_is2 ? c2b[oc] : c1b[oc];
    wire last_k = (kh==2'd2)&&(kw==2'd2)&&(ic==cCin-1);
    wire [11:0] c_outaddr = (oh*cW + ow)*cCout + oc;
    // ReLU + 재양자화(>>7) + 16bit saturate
    wire signed [39:0] acc_relu = (acc<0) ? 40'sd0 : acc;
    wire signed [32:0] acc_q7   = acc_relu >>> 7;
    wire signed [15:0] act_sat  = (acc_q7 > 33'sd32767) ? 16'sd32767 : acc_q7[15:0];

    // --- pool 입력 (P1: c1_mem 28x28x5->14x14x5 / P2: c2_mem 14x14x6->7x7x6) ---
    wire        p_is2 = (state==P2);
    wire [4:0]  pInW  = p_is2 ? 5'd14 : 5'd28;     // pool 입력 가로
    wire [4:0]  pOutW = p_is2 ? 5'd7  : 5'd14;     // pool 출력 가로
    wire [2:0]  pC    = p_is2 ? 3'd6  : 3'd5;
    // (oh,ow,oc) = 출력 좌표. 입력 4점 = (2oh+{0,1}, 2ow+{0,1})
    wire [11:0] pa00 = ((2*oh  )*pInW + (2*ow  ))*pC + oc;
    wire [11:0] pa01 = ((2*oh  )*pInW + (2*ow+1))*pC + oc;
    wire [11:0] pa10 = ((2*oh+1)*pInW + (2*ow  ))*pC + oc;
    wire [11:0] pa11 = ((2*oh+1)*pInW + (2*ow+1))*pC + oc;
    function signed [15:0] rd_pool_in(input p2, input [11:0] a);
        rd_pool_in = p2 ? c2_mem[a] : c1_mem[a[11:0]];
    endfunction
    wire signed [15:0] pv00=rd_pool_in(p_is2,pa00), pv01=rd_pool_in(p_is2,pa01),
                       pv10=rd_pool_in(p_is2,pa10), pv11=rd_pool_in(p_is2,pa11);
    wire signed [15:0] pmax01 = (pv00>pv01)?pv00:pv01;
    wire signed [15:0] pmax23 = (pv10>pv11)?pv10:pv11;
    wire signed [15:0] pmax   = (pmax01>pmax23)?pmax01:pmax23;
    wire [11:0] p_outaddr = (oh*pOutW + ow)*pC + oc;

    // --- dense 읽기 (D1: in=p2_mem(294), out16 / D2: in=d1_mem(16), out10) ---
    wire        d_is2 = (state==D2_INIT)||(state==D2_MAC)||(state==D2_WB);
    wire [12:0] dN    = d_is2 ? 13'd16  : 13'd294;
    wire signed [15:0] d_in = d_is2 ? d1_mem[ii[3:0]] : p2_mem[ii[8:0]];
    wire [12:0] dw_addr = jj*dN + ii;
    wire signed [7:0]  dw_val = d_is2 ? d2w[dw_addr[7:0]] : d1w[dw_addr];
    wire signed [31:0] db_val = d_is2 ? d2b[jj[3:0]] : d1b[jj[3:0]];
    wire d_last = (ii==dN-1);

    // argmax
    reg [3:0] arg_i;
    reg signed [31:0] best_val;
    reg [3:0] best_idx;

    // ---------------- 메인 FSM ----------------
    integer k;
    always @(posedge clk) begin
        if (rst) begin
            state<=S_IDLE; done<=0; pred_class<=0;
        end else begin
            case (state)
            // -------- 대기 --------
            S_IDLE: if (start) begin
                        done<=0; oh<=0; ow<=0; oc<=0; kh<=0; kw<=0; ic<=0;
                        acc <= c1b[0];           // 첫 출력원소 = bias로 시작
                        state<=C1_MAC;
                    end
            // ======== CONV1 / CONV2 공용 흐름 ========
            C1_INIT, C2_INIT: begin
                        kh<=0; kw<=0; ic<=0;
                        acc <= cb_val;
                        state <= c_is2 ? C2_MAC : C1_MAC;
                    end
            C1_MAC, C2_MAC: begin
                        acc <= acc + in_val * cw_val;   // signed MAC
                        if (last_k) state <= c_is2 ? C2_WB : C1_WB;
                        else begin
                            // 커널 인덱스 증가 ic->kw->kh
                            if (ic==cCin-1) begin ic<=0;
                                if (kw==2) begin kw<=0; kh<=kh+1; end
                                else kw<=kw+1;
                            end else ic<=ic+1;
                        end
                    end
            C1_WB: begin
                        c1_mem[c_outaddr] <= act_sat;
                        // 출력 인덱스 증가 oc->ow->oh
                        if (oc==cCout-1) begin oc<=0;
                            if (ow==cW-1) begin ow<=0;
                                if (oh==cH-1) begin // conv1 끝 -> pool1
                                    oh<=0; ow<=0; oc<=0; state<=P1;
                                end else begin oh<=oh+1; state<=C1_INIT; end
                            end else begin ow<=ow+1; state<=C1_INIT; end
                        end else begin oc<=oc+1; state<=C1_INIT; end
                    end
            C2_WB: begin
                        c2_mem[c_outaddr] <= act_sat;
                        if (oc==cCout-1) begin oc<=0;
                            if (ow==cW-1) begin ow<=0;
                                if (oh==cH-1) begin oh<=0; ow<=0; oc<=0; state<=P2;
                                end else begin oh<=oh+1; state<=C2_INIT; end
                            end else begin ow<=ow+1; state<=C2_INIT; end
                        end else begin oc<=oc+1; state<=C2_INIT; end
                    end
            // ======== POOL1 ======== (14x14x5)
            P1: begin
                        p1_mem[p_outaddr] <= pmax;
                        if (oc==pC-1) begin oc<=0;
                            if (ow==pOutW-1) begin ow<=0;
                                if (oh==(5'd14-1)) begin // pool1 끝 -> conv2
                                    oh<=0; ow<=0; oc<=0; state<=C2_INIT;
                                end else oh<=oh+1;
                            end else ow<=ow+1;
                        end else oc<=oc+1;
                    end
            // ======== POOL2 ======== (7x7x6)
            P2: begin
                        p2_mem[p_outaddr] <= pmax;
                        if (oc==pC-1) begin oc<=0;
                            if (ow==pOutW-1) begin ow<=0;
                                if (oh==(5'd7-1)) begin // pool2 끝 -> dense1
                                    jj<=0; ii<=0; acc<=d1b[0]; state<=D1_MAC;
                                end else oh<=oh+1;
                            end else ow<=ow+1;
                        end else oc<=oc+1;
                    end
            // ======== DENSE1 / DENSE2 공용 ========
            D1_INIT, D2_INIT: begin
                        ii<=0; acc<=db_val; state<= d_is2 ? D2_MAC : D1_MAC;
                    end
            D1_MAC, D2_MAC: begin
                        acc <= acc + d_in * dw_val;
                        if (d_last) state <= d_is2 ? D2_WB : D1_WB;
                        else ii<=ii+1;
                    end
            D1_WB: begin
                        d1_mem[jj[3:0]] <= act_sat;       // ReLU+>>7
                        if (jj==15) begin jj<=0; ii<=0; acc<=d2b[0]; state<=D2_MAC; end
                        else begin jj<=jj+1; state<=D1_INIT; end
                    end
            D2_WB: begin
                        logit_mem[jj[3:0]] <= acc[31:0];  // 마지막층: ReLU/시프트 없이 Q14
                        if (jj==9) begin
                            arg_i<=1; best_val<=logit_mem[0]; best_idx<=0; state<=ARG;
                        end else begin jj<=jj+1; state<=D2_INIT; end
                    end
            // ======== ARGMAX ========
            ARG: begin
                        if (arg_i==10) begin
                            pred_class<=best_idx; done<=1; state<=S_IDLE;
                        end else begin
                            if (logit_mem[arg_i] > best_val) begin
                                best_val<=logit_mem[arg_i]; best_idx<=arg_i;
                            end
                            arg_i<=arg_i+1;
                        end
                    end
            endcase
        end
    end
endmodule
