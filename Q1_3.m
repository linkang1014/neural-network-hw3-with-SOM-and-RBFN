%%
%INPUT
clear all;
close all;
x = -1.6:0.08:1.6;%training set
x_test = -1.6:0.01:1.6;%test set
random_noise = randn(size(x)) ;
y = 1.2*sin(pi*x)-cos(2.4*pi*x)+0.3*random_noise;%target values of training set
y_test = 1.2*sin(pi*x_test)-cos(2.4*pi*x_test);%true value of test set
i = 10;%the value of the regularization factor
%%
%CACULATION
function_r = exp(-(dist(x)).^2/0.02);%Gaussian Functions
w = inv(function_r'*function_r+i*eye(size(function_r)))*function_r'*y'; %Weight matrix

function_r_test = exp(-(dist(x_test',x)).^2/0.02);%Gaussian Functions
y_test_out = (function_r_test*w)';

evaluate = sum((y_test-y_test_out).^2)/length(x_test);%Evaluate the performance
%%
%OUTPUT
figure;
plot(x_test,y_test,'b-');
hold on;
plot(x_test,y_test_out,'r*');
legend('test true result','RBFN result');
hold off;
discription1 = sprintf('With regularization Factor = %d',i);
title(discription1,'FontSize',20);
discription2 = sprintf('Q1_3_%d.jpg',i);
% saveas(gcf,discription2);
% close;



