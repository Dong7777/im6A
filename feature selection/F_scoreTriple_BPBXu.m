 clc
clear
load('XX1.mat')
[X1,ps] = mapminmax(XX1');
X1=X1';
x1=X1(1:size(X1,1)/2,:);
x2=X1((size(X1,1)/2)+1:end,:);

% x1表示正样本矩阵% x2表示负样本矩阵
x = [x1',x2']';
% x为总的样本
[n1,m1] = size(x1);
[n2,m2] = size(x2);
%找出两个矩阵对应的维度，n个样本，m个维度
aver1 = mean(x1);
aver2 = mean(x2);
aver3 =mean(x);
%对三个矩阵的各个列求平均值，均为1*m的向量
numrator = (aver1-aver3).^2+(aver2-aver3).^2;
%即为计算分子，得到的还是一个1*m的向量
sum_1 = zeros(1,m1);
%赋初值，才能计算分子
for k=1:n1
    chazhi_1 = x1(k,:)-aver1;
    added_1 = chazhi_1 .^2;
    sum_1 = sum_1 + added_1;
end
deno_1 = sum_1/(n1 - 1);
%从for循环到此，得到了分母的前半部分
sum_2 = zeros(1,m2);
for k = 1:n2
    chazhi_2 = x2(k,:)-aver2;
    added_2 = chazhi_2 .^2;
    sum_2 = sum_2 +added_2;
end
deno_2 = sum_2/(n2 - 1);
%得到分母的后半部分
deno = deno_1 + deno_2;
%得到分母
F_1 = numrator ./ deno;
%得到了是未进行排序的F

len = length(F_1);
for k = 1:len
    if isnan(F_1(k))
        F_1(k) = -1;
    end
end
% 去除了F_1中值为NAN的值
[F_2,ind] = sort(F_1,'descend');
% 即对F_1进行了降序排列，ind是其index
F = [F_2',ind']';

save F
save X1 
save ps