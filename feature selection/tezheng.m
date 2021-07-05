clc
clear;
X=[];
M=csvread('3',0);
[m,n]=size(M);
m=csvread('3',0,0,[0,0,m-1,0]);
for i=3:10 
    fileName = [ num2str(i) ];

     X{i}=csvread(fileName,0,1);
end
for i=15:28
    fileName = [ num2str(i) ];

     X{i}=csvread(fileName,0,1);
end
XX=[m,X{3},X{4},X{5},X{6},X{7},X{8},X{9},X{10},X{15},X{16},X{17},X{18},X{19},X{20},X{21},X{22},X{23},X{24},X{25},X{26},X{27},X{28}];
y=XX(:,1);
XX1=XX(:,2:end);
[inst_norm, settings] = mapminmax(XX1);
save XX1
save y