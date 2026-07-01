// =====================================================================
// npu_top.v - Basys3 top module
// Result packet (46 bytes):
//   0xAA, predicted class, 10 x int32 logits (little endian),
//   uint32 NPU cycle count (little endian)
// =====================================================================
module npu_top #(parameter CLKS_PER_BIT = 868)(
    input  wire clk,
    input  wire btnC,
    input  wire RsRx,
    output wire RsTx,
    output wire [15:0] led,
    output wire [6:0]  seg,
    output wire [3:0]  an
);
    wire rst = btnC;

    // UART receiver
    wire [7:0] rx_data;
    wire rx_valid;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) U_RX (
        .clk(clk),
        .rx(RsRx),
        .data(rx_data),
        .valid(rx_valid)
    );

    // Image loader
    reg [9:0] rx_cnt;
    reg       core_start;
    reg       img_we;
    reg [9:0] img_waddr;
    reg [7:0] img_wdata;

    always @(posedge clk) begin
        core_start <= 1'b0;
        img_we <= 1'b0;

        if (rst) begin
            rx_cnt <= 10'd0;
            img_waddr <= 10'd0;
            img_wdata <= 8'd0;
        end else if (rx_valid) begin
            img_we <= 1'b1;
            img_waddr <= rx_cnt;
            img_wdata <= rx_data;

            if (rx_cnt == 10'd783) begin
                rx_cnt <= 10'd0;
                core_start <= 1'b1;
            end else begin
                rx_cnt <= rx_cnt + 1'b1;
            end
        end
    end

    // NPU core
    wire core_done;
    wire [3:0] core_pred;
    reg  [3:0] rd_idx;
    wire signed [31:0] rd_logit;

    npu_core CORE (
        .clk(clk),
        .rst(rst),
        .start(core_start),
        .img_we(img_we),
        .img_waddr(img_waddr),
        .img_wdata(img_wdata),
        .done(core_done),
        .pred_class(core_pred),
        .rd_idx(rd_idx),
        .rd_logit(rd_logit)
    );

    // Rising-edge detector for core_done
    reg core_done_d;
    always @(posedge clk) begin
        if (rst)
            core_done_d <= 1'b0;
        else
            core_done_d <= core_done;
    end

    wire done_rise = core_done & ~core_done_d;

    // Pure NPU cycle counter
    reg        cycle_active;
    reg [31:0] cycle_counter;
    reg [31:0] inference_cycles;

    always @(posedge clk) begin
        if (rst) begin
            cycle_active <= 1'b0;
            cycle_counter <= 32'd0;
            inference_cycles <= 32'd0;
        end else if (core_start) begin
            cycle_active <= 1'b1;
            cycle_counter <= 32'd0;
        end else if (cycle_active) begin
            cycle_counter <= cycle_counter + 1'b1;

            if (done_rise) begin
                cycle_active <= 1'b0;
                inference_cycles <= cycle_counter + 1'b1;
            end
        end
    end

    // UART transmitter
    wire [7:0] tx_data;
    reg tx_start;
    wire tx_busy;
    wire tx_done;

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) U_TX (
        .clk(clk),
        .start(tx_start),
        .data(tx_data),
        .tx(RsTx),
        .busy(tx_busy),
        .done(tx_done)
    );

    localparam [1:0] SD_IDLE = 2'd0,
                     SD_LOAD = 2'd1,
                     SD_WAIT = 2'd2;

    reg [1:0] sd_state;
    reg [5:0] byte_idx;

    wire [3:0] lg_idx = (byte_idx - 6'd2) >> 2;
    wire [1:0] lg_byte = (byte_idx - 6'd2) & 2'b11;
    wire [1:0] cyc_byte = byte_idx - 6'd42;

    reg [7:0] send_byte;

    always @(*) begin
        rd_idx = 4'd0;
        send_byte = 8'd0;

        if ((byte_idx >= 6'd2) && (byte_idx <= 6'd41))
            rd_idx = lg_idx;

        case (byte_idx)
            6'd0: send_byte = 8'hAA;
            6'd1: send_byte = {4'd0, core_pred};
            6'd42, 6'd43, 6'd44, 6'd45:
                send_byte = inference_cycles[8*cyc_byte +: 8];
            default:
                send_byte = rd_logit[8*lg_byte +: 8];
        endcase
    end

    assign tx_data = send_byte;

    always @(posedge clk) begin
        tx_start <= 1'b0;

        if (rst) begin
            sd_state <= SD_IDLE;
            byte_idx <= 6'd0;
        end else begin
            case (sd_state)
                SD_IDLE: begin
                    if (done_rise) begin
                        byte_idx <= 6'd0;
                        sd_state <= SD_LOAD;
                    end
                end

                SD_LOAD: begin
                    if (!tx_busy) begin
                        tx_start <= 1'b1;
                        sd_state <= SD_WAIT;
                    end
                end

                SD_WAIT: begin
                    if (tx_done) begin
                        if (byte_idx == 6'd45) begin
                            sd_state <= SD_IDLE;
                        end else begin
                            byte_idx <= byte_idx + 1'b1;
                            sd_state <= SD_LOAD;
                        end
                    end
                end

                default: sd_state <= SD_IDLE;
            endcase
        end
    end

    // Keep seven-segment prediction display; disable all LEDs.
    seven_seg DISP (
        .digit(core_pred),
        .seg(seg),
        .an(an)
    );

    assign led = 16'd0;

endmodule
