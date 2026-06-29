// Basys3 7세그먼트: 맨 오른쪽 자리(AN0)에 숫자 1개(0~9) 표시, 나머지 끔
// seg[0]=a ... seg[6]=g, active-low (0=켜짐). an active-low.
module seven_seg(
    input  wire [3:0] digit,    // 0~9
    output reg  [6:0] seg,
    output wire [3:0] an
);
    assign an = 4'b1110;         // AN0만 켜기 (active-low)
    reg [6:0] on;                // bit0=a..bit6=g, 1=켜짐
    always @(*) begin
        case (digit)
            4'd0: on = 7'b0111111;
            4'd1: on = 7'b0000110;
            4'd2: on = 7'b1011011;
            4'd3: on = 7'b1001111;
            4'd4: on = 7'b1100110;
            4'd5: on = 7'b1101101;
            4'd6: on = 7'b1111101;
            4'd7: on = 7'b0000111;
            4'd8: on = 7'b1111111;
            4'd9: on = 7'b1101111;
            default: on = 7'b0000000;
        endcase
        seg = ~on;               // active-low
    end
endmodule
