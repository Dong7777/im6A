
clc
clear
load('X1.mat')
load('F.mat')
X=zeros(9210,1025);
for i=1:1025
    k=F(2,i);
    X(:,i)=X1(:,k);
end
 save X