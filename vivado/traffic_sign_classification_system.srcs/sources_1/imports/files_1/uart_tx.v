// UART 송신기 8N1 (LSB first)
module uart_tx #(parameter CLKS_PER_BIT = 868)(
    input  wire clk,
    input  wire start,         // 1클럭 펄스 -> 전송 시작
    input  wire [7:0] data,
    output reg  tx,            // 직렬 출력
    output reg  busy,          // 전송 중
    output reg  done           // 1클럭 펄스 (전송 완료)
);
    localparam IDLE=0, START=1, DATA=2, STOP=3;
    reg [1:0] state = IDLE;
    reg [15:0] cnt = 0;
    reg [2:0]  bidx = 0;
    reg [7:0]  shift = 0;
    initial begin tx=1'b1; busy=0; done=0; end

    always @(posedge clk) begin
        done <= 1'b0;
        case (state)
            IDLE: begin tx<=1'b1; cnt<=0; bidx<=0;
                if (start) begin shift<=data; busy<=1'b1; state<=START; end
                else busy<=1'b0;
            end
            START: begin tx<=1'b0;                 // 시작비트
                if (cnt==CLKS_PER_BIT-1) begin cnt<=0; state<=DATA; end
                else cnt<=cnt+1;
            end
            DATA: begin tx<=shift[bidx];           // LSB first
                if (cnt==CLKS_PER_BIT-1) begin cnt<=0;
                    if (bidx==7) state<=STOP; else bidx<=bidx+1;
                end else cnt<=cnt+1;
            end
            STOP: begin tx<=1'b1;                   // 정지비트
                if (cnt==CLKS_PER_BIT-1) begin
                    cnt<=0; busy<=1'b0; done<=1'b1; state<=IDLE;
                end else cnt<=cnt+1;
            end
        endcase
    end
endmodule
