%%
%INPUT
clc;
clear all;
close all;
load('mnist_m.mat');
Index_train = find(train_classlabel==1|train_classlabel==8);
Index_test = find(test_classlabel==1|test_classlabel==8);
for i=1:203
    train_classlabel(Index_train(i)) = 1;
end
for i=1:1000
    if train_classlabel(i)~=1
        train_classlabel(i) = 0;
    end
end
for i=1:48
    test_classlabel(Index_train(i)) = 1;
end
for i=1:250
    if test_classlabel(i)~=1
        test_classlabel(i) = 0;
    end
end
train_data = double(train_data);
test_data = double(test_data);
train_classlabel = double(train_classlabel);
test_classlabel = double(test_classlabel);

train_data_mean=mean(mean(train_data,2));
sigma=std2(train_data);
train_data=(train_data-train_data_mean)./sigma;
test_data=(test_data-train_data_mean)./sigma;

temp = randperm(size(train_data,2));
mu = train_data(:,temp(1:100));%fixed centres selected at random
dm = 150;
%%
%CACULATION
function_RBF = exp(-size(mu,2)/(dm^2)*(dist(train_data',mu)).^2);%RBF Functions
w = pinv(function_RBF'*function_RBF)*function_RBF'*train_classlabel';%Weight matrix
function_RBF_test = exp(-size(mu,2)/(dm^2)*(dist(test_data',mu)).^2);
%%
%Evaluate
TrPred = function_RBF*w;
TePred = function_RBF_test*w;

TrAcc = zeros(1,1000);
TeAcc = zeros(1,1000);
thr = zeros(1,1000);
TrN = length(train_classlabel);
TeN = length(test_classlabel);
for i = 1:1000
t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
thr(i) = t;
TrAcc(i) = (sum(train_classlabel(TrPred<t)==0) + sum(train_classlabel(TrPred>=t)==1)) / TrN;
TeAcc(i) = (sum(test_classlabel(TePred<t)==0) + sum(test_classlabel(TePred>=t)==1)) / TeN;
end
plot(thr,TrAcc,'.-',thr,TeAcc,'^-');
legend('tr','te');
hold on;
discription1 = sprintf('RBFN fixed centers width = %d',dm);
title(discription1,'FontSize',20);
xlabel('Thresholds','FontSize',16);
ylabel('Accuracy','FontSize',16);
discription2 = sprintf('Q2_2_%d.jpg',dm);
saveas(gcf,discription2);
close;

