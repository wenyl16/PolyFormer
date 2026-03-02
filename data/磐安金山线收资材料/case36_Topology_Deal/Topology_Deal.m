clc;
clear;

%%bus-36 model

base = 1000;

main_line_par1 = [261.2, 577.6, 615.7] ./ base;
main_line_par2 = [265.07, 108.8, 562, 140.6, 120.6] ./ base;
main_line_par1_sub = [703.2 , 82.85] ./ base;
main_line_par2_sub = [420.7, 102.5, 218.17, 224.95] ./ base;

sub_line_1_par2 = [206.295, 527.33, 19.7027] ./ base;
sub_line_2_par3 = [55.966, 353.8, 1291.7, 714, 1157.27, 1399.12] ./ base;
sub_line_3_par3 = [46.422, 36.956, 198, 43.76, 141.75, 61.62, 19.99] ./ base;
sub_line_4_par3 = [44.21, 55.407, 43.767, 44.026, 143.23] ./ base;

%par1 : r = 0.226/km ; x = 0.334/km
%par2 : r = 0.183/km ; x = 0.327/km
%par3 : r = 0.235/km ; x = 0.377/km

%parameter setting
par1_r = 0.226 / 2; par1_x = 0.334 / 2;
par2_r = 0.183 / 2; par2_x = 0.327 / 2;
par3_r = 0.235 / 2; par3_x = 0.377 / 2;

main_line_r = zeros(14,1);
main_line_x = zeros(14,1);
sub_line_1_r = zeros(3,1);
sub_line_1_x = zeros(3,1);
sub_line_2_r = zeros(6,1);
sub_line_2_x = zeros(6,1);
sub_line_3_r = zeros(7,1);
sub_line_3_x = zeros(7,1);
sub_line_4_r = zeros(5,1);
sub_line_4_x = zeros(5,1);

for i = 1:3
    main_line_r(i) = par1_r * main_line_par1(i);
    main_line_x(i) = par1_x * main_line_par1(i);
end
for i = 1:5
    main_line_r(i + 3) = par2_r * main_line_par2(i);
    main_line_x(i + 3) = par2_x * main_line_par2(i);
end
for i =1:2
    main_line_r(i + 8) = par1_r * main_line_par1_sub(i);
    main_line_x(i + 8) = par1_x * main_line_par1_sub(i);
end
for i = 1:4
    main_line_r(i + 10) = par2_r * main_line_par2_sub(i);
    main_line_x(i + 10) = par2_x * main_line_par2_sub(i);
end
for i = 1:3
    sub_line_1_r(i) = par2_r * sub_line_1_par2(i);
    sub_line_1_x(i) = par2_x * sub_line_1_par2(i);
end
for i = 1:6
    sub_line_2_r(i) = par3_r * sub_line_2_par3(i);
    sub_line_2_x(i) = par3_x * sub_line_2_par3(i);
end
for i = 1:7
    sub_line_3_r(i) = par3_r * sub_line_3_par3(i);
    sub_line_3_x(i) = par3_x * sub_line_3_par3(i);
end
for i = 1:5
    sub_line_4_r(i) = par3_r * sub_line_4_par3(i);
    sub_line_4_x(i) = par3_x * sub_line_4_par3(i);
end
%%