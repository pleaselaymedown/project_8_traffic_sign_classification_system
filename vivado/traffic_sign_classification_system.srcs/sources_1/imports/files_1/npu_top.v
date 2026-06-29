// =====================================================================
//  npu_top.v  -  Basys3 최상위 모듈
//  PC --(UART 784바이트 이미지)--> FPGA --(추론)--> --(결과패킷)--> PC
//  결과 패킷(42바이트): 0xAA, 예측클래스, logit0..9(각 int32 little-endian)
// =====================================================================
module npu_top #(parameter CLKS_PER_BIT = 868)(   // 100MHz/115200=868
    input  wire clk,
    input  wire btnC,            // 리셋
    input  wire RsRx,            // PC -> FPGA
    output wire RsTx,            // FPGA -> PC
    output reg  [15:0] led,
    output wire [6:0]  seg,
    output wire [3:0]  an
);
    wire rst = btnC;

    // ---------- UART 수신 ----------
    wire [7:0] rx_data; wire rx_valid;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) U_RX
        (.clk(clk), .rx(RsRx), .data(rx_data), .valid(rx_valid));

    // ---------- 이미지 로더 FSM ----------
    reg [9:0] rx_cnt = 0;
    reg       core_start = 0;
    reg       img_we = 0; reg [9:0] img_waddr = 0; reg [7:0] img_wdata = 0;
    always @(posedge clk) begin
        core_start <= 1'b0;
        img_we     <= 1'b0;
        if (rst) begin rx_cnt <= 0; end
        else if (rx_valid) begin
            img_we    <= 1'b1;          // 받은 바이트를 img_mem에 기록
            img_waddr <= rx_cnt;
            img_wdata <= rx_data;
            if (rx_cnt == 10'd783) begin
                rx_cnt     <= 0;
                core_start <= 1'b1;     // 784바이트 다 받으면 추론 시작
            end else rx_cnt <= rx_cnt + 1;
        end
    end

    // ---------- NPU 코어 ----------
    wire core_done; wire [3:0] core_pred;
    reg  [3:0] rd_idx; wire signed [31:0] rd_logit;
    npu_core CORE(
        .clk(clk), .rst(rst), .start(core_start),
        .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .done(core_done), .pred_class(core_pred),
        .rd_idx(rd_idx), .rd_logit(rd_logit));

    // ---------- 결과 송신 FSM ----------
    wire [7:0] tx_data; reg tx_start; wire tx_busy, tx_done;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) U_TX
        (.clk(clk), .start(tx_start), .data(tx_data),
         .tx(RsTx), .busy(tx_busy), .done(tx_done));

    localparam SD_IDLE=0, SD_LOAD=1, SD_WAIT=2;
    reg [1:0] sd_state = SD_IDLE;
    reg [5:0] byte_idx = 0;          // 0~41
    reg core_done_d = 0;
    always @(posedge clk) core_done_d <= core_done;
    wire done_rise = core_done & ~core_done_d;

    // 보낼 바이트 선택 (조합)
    // byte 0: 0xAA / byte 1: class / byte 2~41: logit[(i-2)/4] 의 little-endian
    wire [3:0]  lg_idx  = (byte_idx-2) >> 2;     // 어떤 logit
    wire [1:0]  lg_byte = (byte_idx-2) & 2'b11;  // 그 logit의 몇번째 바이트
    reg  [7:0] send_byte;
    always @(*) begin
        rd_idx = lg_idx;
        case (byte_idx)
            6'd0: send_byte = 8'hAA;
            6'd1: send_byte = {4'd0, core_pred};
            default: send_byte = rd_logit[8*lg_byte +: 8];  // LE 바이트 추출
        endcase
    end
    assign tx_data = send_byte;

    always @(posedge clk) begin
        tx_start <= 1'b0;
        if (rst) begin sd_state <= SD_IDLE; byte_idx <= 0; end
        else case (sd_state)
            SD_IDLE: if (done_rise) begin byte_idx <= 0; sd_state <= SD_LOAD; end
            SD_LOAD: if (!tx_busy) begin tx_start <= 1'b1; sd_state <= SD_WAIT; end
            SD_WAIT: if (tx_done) begin
                        if (byte_idx == 6'd41) sd_state <= SD_IDLE;   // 42바이트 끝
                        else begin byte_idx <= byte_idx + 1; sd_state <= SD_LOAD; end
                     end
        endcase
    end

    // ---------- 표시 ----------
    seven_seg DISP(.digit(core_pred), .seg(seg), .an(an));
    always @(posedge clk) begin
        led <= 16'd0;
        led[core_pred] <= 1'b1;      // 예측 클래스 자리 LED 점등 (LD0~LD9)
        led[15]        <= core_done; // 추론 완료 표시
    end
endmodule
