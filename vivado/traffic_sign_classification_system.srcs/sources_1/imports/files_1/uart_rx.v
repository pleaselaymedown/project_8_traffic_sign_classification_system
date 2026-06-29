// UART 수신기 8N1 (LSB first)
module uart_rx #(parameter CLKS_PER_BIT = 868)(
    input  wire clk,
    input  wire rx,            // 직렬 입력
    output reg  [7:0] data,    // 수신 바이트
    output reg  valid          // 1클럭 펄스
);
    localparam IDLE=0, START=1, DATA=2, STOP=3;
    reg [1:0] state = IDLE;
    reg [15:0] cnt = 0;        // 클럭 카운터
    reg [2:0]  bidx = 0;       // 비트 인덱스 0~7
    reg rx_d1=1, rx_d2=1;      // 입력 동기화(메타스테이블 방지)
    always @(posedge clk) begin rx_d1<=rx; rx_d2<=rx_d1; end

    always @(posedge clk) begin
        valid <= 1'b0;
        case (state)
            IDLE: begin cnt<=0; bidx<=0;
                if (rx_d2==1'b0) state<=START;   // 시작비트(low) 감지
            end
            START: begin                          // 비트 중앙까지 대기
                if (cnt==(CLKS_PER_BIT-1)/2) begin
                    if (rx_d2==1'b0) begin cnt<=0; state<=DATA; end
                    else state<=IDLE;             // 노이즈
                end else cnt<=cnt+1;
            end
            DATA: begin
                if (cnt==CLKS_PER_BIT-1) begin
                    cnt<=0; data[bidx]<=rx_d2;     // LSB first 샘플
                    if (bidx==7) state<=STOP; else bidx<=bidx+1;
                end else cnt<=cnt+1;
            end
            STOP: begin
                if (cnt==CLKS_PER_BIT-1) begin
                    valid<=1'b1; state<=IDLE;       // 정지비트 후 valid
                end else cnt<=cnt+1;
            end
        endcase
    end
endmodule
