// =====================================================================
// npu_core.v - timing-improved sequential CNN inference core
//
// Main timing changes compared with the original version:
//   1) Large activation memories and weight ROMs use synchronous reads.
//   2) Memory read and MAC are separated into different FSM states.
//   3) Max pooling reads one pixel at a time instead of four in parallel.
//   4) Dense weight addresses use a running counter instead of jj*dN+ii.
//
// Model:
// 28x28x1 -> Conv5 -> Pool -> Conv6 -> Pool -> Dense16 -> Dense10
// Fixed point:
// input/weight Q7, bias/accumulator Q14, hidden activation requantized to Q7.
// =====================================================================
module npu_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,

    input  wire        img_we,
    input  wire [9:0]  img_waddr,
    input  wire [7:0]  img_wdata,

    output reg         done,
    output reg  [3:0]  pred_class,

    input  wire [3:0]  rd_idx,
    output wire signed [31:0] rd_logit
);

    // -----------------------------------------------------------------
    // Memories
    // -----------------------------------------------------------------
    // Synchronous read coding style is used so Vivado can infer BRAM/ROM
    // instead of creating very large asynchronous LUT multiplexers.
    (* ram_style = "block" *) reg        [7:0]  img_mem [0:783];
    (* ram_style = "block" *) reg signed [15:0] c1_mem  [0:3919];
    (* ram_style = "block" *) reg signed [15:0] p1_mem  [0:979];
    (* ram_style = "block" *) reg signed [15:0] c2_mem  [0:1175];
    (* ram_style = "block" *) reg signed [15:0] p2_mem  [0:293];
    (* ram_style = "distributed" *) reg signed [15:0] d1_mem [0:15];
    reg signed [31:0] logit_mem [0:9];

    (* rom_style = "block" *) reg signed [7:0] c1w [0:44];
    (* rom_style = "block" *) reg signed [7:0] c2w [0:269];
    (* rom_style = "block" *) reg signed [7:0] d1w [0:4703];
    (* rom_style = "block" *) reg signed [7:0] d2w [0:159];

    reg signed [31:0] c1b [0:4];
    reg signed [31:0] c2b [0:5];
    reg signed [31:0] d1b [0:15];
    reg signed [31:0] d2b [0:9];

    initial begin
        $readmemh("conv1_w.mem",  c1w);
        $readmemh("conv1_b.mem",  c1b);
        $readmemh("conv2_w.mem",  c2w);
        $readmemh("conv2_b.mem",  c2b);
        $readmemh("dense1_w.mem", d1w);
        $readmemh("dense1_b.mem", d1b);
        $readmemh("dense2_w.mem", d2w);
        $readmemh("dense2_b.mem", d2b);
    end

    assign rd_logit = (rd_idx < 4'd10) ? logit_mem[rd_idx] : 32'sd0;

    // -----------------------------------------------------------------
    // FSM states
    // -----------------------------------------------------------------
    localparam [4:0]
        S_IDLE  = 5'd0,

        C1_INIT = 5'd1,
        C1_READ = 5'd2,
        C1_MUL  = 5'd3,
        C1_ACC  = 5'd4,
        C1_WB   = 5'd5,

        P1_READ = 5'd6,
        P1_ACC  = 5'd7,
        P1_WB   = 5'd8,

        C2_INIT = 5'd9,
        C2_READ = 5'd10,
        C2_MUL  = 5'd11,
        C2_ACC  = 5'd12,
        C2_WB   = 5'd13,

        P2_READ = 5'd14,
        P2_ACC  = 5'd15,
        P2_WB   = 5'd16,

        D1_INIT = 5'd17,
        D1_READ = 5'd18,
        D1_MUL  = 5'd19,
        D1_ACC  = 5'd20,
        D1_WB   = 5'd21,

        D2_INIT = 5'd22,
        D2_READ = 5'd23,
        D2_MUL  = 5'd24,
        D2_ACC  = 5'd25,
        D2_WB   = 5'd26,

        ARG     = 5'd27;

    reg [4:0] state;

    wire conv2_active =
        (state == C2_INIT) || (state == C2_READ) ||
        (state == C2_MUL)  || (state == C2_ACC)  ||
        (state == C2_WB);

    wire pool2_active =
        (state == P2_READ) || (state == P2_ACC) || (state == P2_WB);

    wire dense2_active =
        (state == D2_INIT) || (state == D2_READ) ||
        (state == D2_MUL)  || (state == D2_ACC)  ||
        (state == D2_WB);

    // -----------------------------------------------------------------
    // Common counters/registers
    // -----------------------------------------------------------------
    reg [4:0] oh;
    reg [4:0] ow;
    reg [2:0] oc;
    reg [1:0] kh;
    reg [1:0] kw;
    reg [2:0] ic;

    reg [12:0] jj;
    reg [12:0] ii;

    reg signed [39:0] acc;
    reg signed [23:0] mac_product;

    // Running addresses remove large output/dense address multipliers.
    reg [11:0] conv_outaddr;
    reg [11:0] pool_outaddr;
    reg [12:0] conv_waddr;
    reg [12:0] dense_waddr;

    // -----------------------------------------------------------------
    // Convolution address generation
    // -----------------------------------------------------------------
    wire [4:0] conv_h    = conv2_active ? 5'd14 : 5'd28;
    wire [4:0] conv_w    = conv2_active ? 5'd14 : 5'd28;
    wire [2:0] conv_cin  = conv2_active ? 3'd5  : 3'd1;
    wire [2:0] conv_cout = conv2_active ? 3'd6  : 3'd5;

    wire signed [6:0] conv_ih_s =
        $signed({2'b00, oh}) + $signed({5'b00000, kh}) - 7'sd1;
    wire signed [6:0] conv_iw_s =
        $signed({2'b00, ow}) + $signed({5'b00000, kw}) - 7'sd1;

    wire conv_in_range =
        (conv_ih_s >= 0) &&
        (conv_iw_s >= 0) &&
        (conv_ih_s < $signed({2'b00, conv_h})) &&
        (conv_iw_s < $signed({2'b00, conv_w}));

    wire [5:0] conv_ih_u = conv_ih_s[5:0];
    wire [5:0] conv_iw_u = conv_iw_s[5:0];

    // These are constant multiplications after synthesis.
    wire [10:0] c1_in_addr_calc =
        (conv_ih_u * 6'd28) + conv_iw_u;

    wire [10:0] c2_pixel_index =
        (conv_ih_u * 5'd14) + conv_iw_u;

    wire [10:0] c2_in_addr_calc =
        (c2_pixel_index * 3'd5) + ic;

    wire [10:0] conv_in_addr_calc =
        conv2_active ? c2_in_addr_calc : c1_in_addr_calc;

    wire [10:0] conv_in_addr_safe =
        conv_in_range ? conv_in_addr_calc : 11'd0;

    wire conv_last_k =
        (kh == 2'd2) &&
        (kw == 2'd2) &&
        (ic == conv_cin - 1'b1);

    // Weight base for each output channel.
    // Conv1: oc*9, Conv2: oc*45.
    wire [12:0] c1_weight_base =
        ({10'd0, oc} << 3) + {10'd0, oc};

    wire [12:0] c2_weight_base =
        ({10'd0, oc} << 5) +
        ({10'd0, oc} << 3) +
        ({10'd0, oc} << 2) +
        {10'd0, oc};

    reg conv_in_range_q;

    // -----------------------------------------------------------------
    // Pooling address generation: one memory read at a time
    // -----------------------------------------------------------------
    reg [1:0] pool_step;
    reg signed [15:0] pool_max;

    wire [5:0] pool_row = ({1'b0, oh} << 1) + pool_step[1];
    wire [5:0] pool_col = ({1'b0, ow} << 1) + pool_step[0];

    wire [11:0] p1_pixel_index =
        (pool_row * 6'd28) + pool_col;
    wire [11:0] p1_addr_calc =
        (p1_pixel_index * 3'd5) + oc;

    wire [11:0] p2_pixel_index =
        (pool_row * 5'd14) + pool_col;
    wire [11:0] p2_addr_calc =
        (p2_pixel_index * 3'd6) + oc;

    wire [11:0] pool_in_addr =
        pool2_active ? p2_addr_calc : p1_addr_calc;

    wire [4:0] pool_out_w = pool2_active ? 5'd7 : 5'd14;
    wire [2:0] pool_channels = pool2_active ? 3'd6 : 3'd5;

    // -----------------------------------------------------------------
    // Registered synchronous memory/ROM outputs
    // -----------------------------------------------------------------
    reg        [7:0] img_q;
    reg signed [15:0] p1_conv_q;
    reg signed [15:0] c1_pool_q;
    reg signed [15:0] c2_pool_q;
    reg signed [15:0] p2_dense_q;
    reg signed [15:0] d1_dense_q;

    reg signed [7:0] c1w_q;
    reg signed [7:0] c2w_q;
    reg signed [7:0] d1w_q;
    reg signed [7:0] d2w_q;

    wire [9:0] img_raddr =
        ((state == C1_READ) && conv_in_range) ?
        conv_in_addr_safe[9:0] : 10'd0;

    wire [9:0] p1_conv_raddr =
        ((state == C2_READ) && conv_in_range) ?
        conv_in_addr_safe[9:0] : 10'd0;

    wire [11:0] c1_pool_raddr =
        (state == P1_READ) ? pool_in_addr : 12'd0;

    wire [10:0] c2_pool_raddr =
        (state == P2_READ) ? pool_in_addr[10:0] : 11'd0;

    wire [8:0] p2_dense_raddr =
        (state == D1_READ) ? ii[8:0] : 9'd0;

    wire [3:0] d1_dense_raddr =
        (state == D2_READ) ? ii[3:0] : 4'd0;

    wire [5:0] c1w_raddr =
        (state == C1_READ) ? conv_waddr[5:0] : 6'd0;

    wire [8:0] c2w_raddr =
        (state == C2_READ) ? conv_waddr[8:0] : 9'd0;

    wire [12:0] d1w_raddr =
        (state == D1_READ) ? dense_waddr : 13'd0;

    wire [7:0] d2w_raddr =
        (state == D2_READ) ? dense_waddr[7:0] : 8'd0;

    // Image RAM: UART write port + NPU synchronous read port.
    always @(posedge clk) begin
        if (img_we)
            img_mem[img_waddr] <= img_wdata;
        img_q <= img_mem[img_raddr];
    end

    // Intermediate activation memory read ports.
    always @(posedge clk) begin
        p1_conv_q  <= p1_mem[p1_conv_raddr];
        c1_pool_q  <= c1_mem[c1_pool_raddr];
        c2_pool_q  <= c2_mem[c2_pool_raddr];
        p2_dense_q <= p2_mem[p2_dense_raddr];
        d1_dense_q <= d1_mem[d1_dense_raddr];
    end

    // Weight ROM synchronous read ports.
    always @(posedge clk) begin
        c1w_q <= c1w[c1w_raddr];
        c2w_q <= c2w[c2w_raddr];
        d1w_q <= d1w[d1w_raddr];
        d2w_q <= d2w[d2w_raddr];
    end

    wire signed [15:0] conv_input_q =
        conv2_active ? p1_conv_q : $signed({8'd0, img_q});

    wire signed [7:0] conv_weight_q =
        conv2_active ? c2w_q : c1w_q;

    wire signed [15:0] pool_input_q =
        pool2_active ? c2_pool_q : c1_pool_q;

    wire signed [15:0] dense_input_q =
        dense2_active ? d1_dense_q : p2_dense_q;

    wire signed [7:0] dense_weight_q =
        dense2_active ? d2w_q : d1w_q;

    wire signed [31:0] conv_bias =
        conv2_active ? c2b[oc] : c1b[oc];

    wire signed [31:0] dense_bias =
        dense2_active ? d2b[jj[3:0]] : d1b[jj[3:0]];

    wire [12:0] dense_n = dense2_active ? 13'd16 : 13'd294;
    wire dense_last = (ii == dense_n - 1'b1);

    // -----------------------------------------------------------------
    // ReLU / requantization
    // -----------------------------------------------------------------
    wire signed [39:0] acc_relu = (acc < 0) ? 40'sd0 : acc;
    wire signed [32:0] acc_q7 = acc_relu >>> 7;
    wire signed [15:0] act_sat =
        (acc_q7 > 33'sd32767) ? 16'sd32767 : acc_q7[15:0];

    // -----------------------------------------------------------------
    // Argmax
    // -----------------------------------------------------------------
    reg [3:0] arg_i;
    reg signed [31:0] best_val;
    reg [3:0] best_idx;

    // -----------------------------------------------------------------
    // Main FSM
    // -----------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            done <= 1'b0;
            pred_class <= 4'd0;

            oh <= 5'd0;
            ow <= 5'd0;
            oc <= 3'd0;
            kh <= 2'd0;
            kw <= 2'd0;
            ic <= 3'd0;
            jj <= 13'd0;
            ii <= 13'd0;
            acc <= 40'sd0;
            mac_product <= 24'sd0;

            conv_outaddr <= 12'd0;
            pool_outaddr <= 12'd0;
            conv_waddr <= 13'd0;
            dense_waddr <= 13'd0;

            pool_step <= 2'd0;
            pool_max <= 16'sd0;
            conv_in_range_q <= 1'b0;

            arg_i <= 4'd0;
            best_val <= 32'sd0;
            best_idx <= 4'd0;
        end else begin
            case (state)
                // -----------------------------------------------------
                // Idle / start
                // -----------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        done <= 1'b0;
                        oh <= 5'd0;
                        ow <= 5'd0;
                        oc <= 3'd0;
                        conv_outaddr <= 12'd0;
                        state <= C1_INIT;
                    end
                end

                // -----------------------------------------------------
                // Conv1
                // -----------------------------------------------------
                C1_INIT: begin
                    kh <= 2'd0;
                    kw <= 2'd0;
                    ic <= 3'd0;
                    conv_waddr <= c1_weight_base;
                    acc <= {{8{c1b[oc][31]}}, c1b[oc]};
                    state <= C1_READ;
                end

                C1_READ: begin
                    conv_in_range_q <= conv_in_range;
                    state <= C1_MUL;
                end

                C1_MUL: begin
                    if (conv_in_range_q)
                        mac_product <= conv_input_q * conv_weight_q;
                    else
                        mac_product <= 24'sd0;
                    state <= C1_ACC;
                end

                C1_ACC: begin
                    acc <= acc + mac_product;

                    if (conv_last_k) begin
                        state <= C1_WB;
                    end else begin
                        conv_waddr <= conv_waddr + 1'b1;

                        if (ic == conv_cin - 1'b1) begin
                            ic <= 3'd0;
                            if (kw == 2'd2) begin
                                kw <= 2'd0;
                                kh <= kh + 1'b1;
                            end else begin
                                kw <= kw + 1'b1;
                            end
                        end else begin
                            ic <= ic + 1'b1;
                        end

                        state <= C1_READ;
                    end
                end

                C1_WB: begin
                    c1_mem[conv_outaddr] <= act_sat;
                    conv_outaddr <= conv_outaddr + 1'b1;

                    if (oc == conv_cout - 1'b1) begin
                        oc <= 3'd0;
                        if (ow == conv_w - 1'b1) begin
                            ow <= 5'd0;
                            if (oh == conv_h - 1'b1) begin
                                oh <= 5'd0;
                                pool_outaddr <= 12'd0;
                                pool_step <= 2'd0;
                                state <= P1_READ;
                            end else begin
                                oh <= oh + 1'b1;
                                state <= C1_INIT;
                            end
                        end else begin
                            ow <= ow + 1'b1;
                            state <= C1_INIT;
                        end
                    end else begin
                        oc <= oc + 1'b1;
                        state <= C1_INIT;
                    end
                end

                // -----------------------------------------------------
                // Pool1: four synchronous reads, one at a time
                // -----------------------------------------------------
                P1_READ: begin
                    state <= P1_ACC;
                end

                P1_ACC: begin
                    if (pool_step == 2'd0)
                        pool_max <= pool_input_q;
                    else if (pool_input_q > pool_max)
                        pool_max <= pool_input_q;

                    if (pool_step == 2'd3) begin
                        state <= P1_WB;
                    end else begin
                        pool_step <= pool_step + 1'b1;
                        state <= P1_READ;
                    end
                end

                P1_WB: begin
                    p1_mem[pool_outaddr] <= pool_max;
                    pool_outaddr <= pool_outaddr + 1'b1;
                    pool_step <= 2'd0;

                    if (oc == pool_channels - 1'b1) begin
                        oc <= 3'd0;
                        if (ow == pool_out_w - 1'b1) begin
                            ow <= 5'd0;
                            if (oh == 5'd13) begin
                                oh <= 5'd0;
                                conv_outaddr <= 12'd0;
                                state <= C2_INIT;
                            end else begin
                                oh <= oh + 1'b1;
                                state <= P1_READ;
                            end
                        end else begin
                            ow <= ow + 1'b1;
                            state <= P1_READ;
                        end
                    end else begin
                        oc <= oc + 1'b1;
                        state <= P1_READ;
                    end
                end

                // -----------------------------------------------------
                // Conv2
                // -----------------------------------------------------
                C2_INIT: begin
                    kh <= 2'd0;
                    kw <= 2'd0;
                    ic <= 3'd0;
                    conv_waddr <= c2_weight_base;
                    acc <= {{8{c2b[oc][31]}}, c2b[oc]};
                    state <= C2_READ;
                end

                C2_READ: begin
                    conv_in_range_q <= conv_in_range;
                    state <= C2_MUL;
                end

                C2_MUL: begin
                    if (conv_in_range_q)
                        mac_product <= conv_input_q * conv_weight_q;
                    else
                        mac_product <= 24'sd0;
                    state <= C2_ACC;
                end

                C2_ACC: begin
                    acc <= acc + mac_product;

                    if (conv_last_k) begin
                        state <= C2_WB;
                    end else begin
                        conv_waddr <= conv_waddr + 1'b1;

                        if (ic == conv_cin - 1'b1) begin
                            ic <= 3'd0;
                            if (kw == 2'd2) begin
                                kw <= 2'd0;
                                kh <= kh + 1'b1;
                            end else begin
                                kw <= kw + 1'b1;
                            end
                        end else begin
                            ic <= ic + 1'b1;
                        end

                        state <= C2_READ;
                    end
                end

                C2_WB: begin
                    c2_mem[conv_outaddr] <= act_sat;
                    conv_outaddr <= conv_outaddr + 1'b1;

                    if (oc == conv_cout - 1'b1) begin
                        oc <= 3'd0;
                        if (ow == conv_w - 1'b1) begin
                            ow <= 5'd0;
                            if (oh == conv_h - 1'b1) begin
                                oh <= 5'd0;
                                pool_outaddr <= 12'd0;
                                pool_step <= 2'd0;
                                state <= P2_READ;
                            end else begin
                                oh <= oh + 1'b1;
                                state <= C2_INIT;
                            end
                        end else begin
                            ow <= ow + 1'b1;
                            state <= C2_INIT;
                        end
                    end else begin
                        oc <= oc + 1'b1;
                        state <= C2_INIT;
                    end
                end

                // -----------------------------------------------------
                // Pool2
                // -----------------------------------------------------
                P2_READ: begin
                    state <= P2_ACC;
                end

                P2_ACC: begin
                    if (pool_step == 2'd0)
                        pool_max <= pool_input_q;
                    else if (pool_input_q > pool_max)
                        pool_max <= pool_input_q;

                    if (pool_step == 2'd3) begin
                        state <= P2_WB;
                    end else begin
                        pool_step <= pool_step + 1'b1;
                        state <= P2_READ;
                    end
                end

                P2_WB: begin
                    p2_mem[pool_outaddr] <= pool_max;
                    pool_outaddr <= pool_outaddr + 1'b1;
                    pool_step <= 2'd0;

                    if (oc == pool_channels - 1'b1) begin
                        oc <= 3'd0;
                        if (ow == pool_out_w - 1'b1) begin
                            ow <= 5'd0;
                            if (oh == 5'd6) begin
                                jj <= 13'd0;
                                ii <= 13'd0;
                                dense_waddr <= 13'd0;
                                state <= D1_INIT;
                            end else begin
                                oh <= oh + 1'b1;
                                state <= P2_READ;
                            end
                        end else begin
                            ow <= ow + 1'b1;
                            state <= P2_READ;
                        end
                    end else begin
                        oc <= oc + 1'b1;
                        state <= P2_READ;
                    end
                end

                // -----------------------------------------------------
                // Dense1
                // -----------------------------------------------------
                D1_INIT: begin
                    ii <= 13'd0;
                    acc <= {{8{d1b[jj[3:0]][31]}}, d1b[jj[3:0]]};
                    state <= D1_READ;
                end

                D1_READ: begin
                    state <= D1_MUL;
                end

                D1_MUL: begin
                    mac_product <= dense_input_q * dense_weight_q;
                    state <= D1_ACC;
                end

                D1_ACC: begin
                    acc <= acc + mac_product;
                    dense_waddr <= dense_waddr + 1'b1;

                    if (dense_last) begin
                        state <= D1_WB;
                    end else begin
                        ii <= ii + 1'b1;
                        state <= D1_READ;
                    end
                end

                D1_WB: begin
                    d1_mem[jj[3:0]] <= act_sat;

                    if (jj == 13'd15) begin
                        jj <= 13'd0;
                        ii <= 13'd0;
                        dense_waddr <= 13'd0;
                        state <= D2_INIT;
                    end else begin
                        jj <= jj + 1'b1;
                        state <= D1_INIT;
                    end
                end

                // -----------------------------------------------------
                // Dense2
                // -----------------------------------------------------
                D2_INIT: begin
                    ii <= 13'd0;
                    acc <= {{8{d2b[jj[3:0]][31]}}, d2b[jj[3:0]]};
                    state <= D2_READ;
                end

                D2_READ: begin
                    state <= D2_MUL;
                end

                D2_MUL: begin
                    mac_product <= dense_input_q * dense_weight_q;
                    state <= D2_ACC;
                end

                D2_ACC: begin
                    acc <= acc + mac_product;
                    dense_waddr <= dense_waddr + 1'b1;

                    if (dense_last) begin
                        state <= D2_WB;
                    end else begin
                        ii <= ii + 1'b1;
                        state <= D2_READ;
                    end
                end

                D2_WB: begin
                    logit_mem[jj[3:0]] <= acc[31:0];

                    if (jj == 13'd9) begin
                        arg_i <= 4'd1;
                        best_val <= logit_mem[0];
                        best_idx <= 4'd0;
                        state <= ARG;
                    end else begin
                        jj <= jj + 1'b1;
                        state <= D2_INIT;
                    end
                end

                // -----------------------------------------------------
                // Argmax
                // -----------------------------------------------------
                ARG: begin
                    if (arg_i == 4'd10) begin
                        pred_class <= best_idx;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        if (logit_mem[arg_i] > best_val) begin
                            best_val <= logit_mem[arg_i];
                            best_idx <= arg_i;
                        end
                        arg_i <= arg_i + 1'b1;
                    end
                end

                default: begin
                    state <= S_IDLE;
                    done <= 1'b0;
                end
            endcase
        end
    end

endmodule
