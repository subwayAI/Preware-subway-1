function fea=tezheng(data)
    %'''data为一维振动信号'''
    x_rms = 0;
    absXbar = 0;
    x_r = 0;
    S = 0;
    K = 0;
    k = 0;
    x_rms = 0;
    fea = [];
    len_ = length(data);
    mean_ = mean(data);%.mean(axis=1)  % 1.均值
    var_ = var(data);%.var(axis=1)  % 2.方差
    std_ = std(data);%.std(axis=1)  % 3.标准差
    max_ = max(data);%.max(axis=1)  % 4.最大值
    min_ = min(data);%.min(axis=1)  % 5.最小值
    x_p = max(abs(max_), abs(min_));  % 6.峰值
    for i =1:len_
        x_rms =x_rms+ data(i) .^ 2;
        absXbar =absXbar + abs(data(i));
        x_r= x_r + sqrt(abs(data(i)));
        S = S+(data(i) - mean_) .^ 3;
        K = K+(data(i) - mean_) .^ 4;
    end
    x_rms = sqrt(x_rms / len_);  % 7.均方根值
    absXbar = absXbar / len_;  % 8.绝对平均值
    x_r = (x_r / len_) .^ 2;  % 9.方根幅值
    W = x_rms / mean_ ; % 10.波形指标
    C = x_p / x_rms ; % 11.峰值指标
    I = x_p / mean_ ; % 12.脉冲指标
    L = x_p / x_r  ;% 13.裕度指标
    S = S / ((len_ - 1) * std_ .^ 3);  % 14.偏斜度
    K = K / ((len_ - 1) * std_ .^ 4) ; % 15.峭度

    fea = [mean_,absXbar,var_,std_,x_r,x_rms,x_p,max_,min_,W,C,I,L,S,K];
  
%