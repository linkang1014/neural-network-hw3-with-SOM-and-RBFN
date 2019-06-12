%%
%Input
clc
clear all;
close all;
load('Digits.mat');
%omit the specific class 1 3
Index_train = find(train_classlabel==1|train_classlabel==3);
Index_test = find(test_classlabel==1|test_classlabel==3);
train_data(:,(Index_train)) = [];
train_classlabel(:,(Index_train)) = [];
test_data(:,(Index_test)) = [];
test_classlabel(:,(Index_test)) = [];
%transform to double
train_data = double(train_data);
train_classlabel = double(train_classlabel);
test_data = double(test_data);
test_classlabel = double(test_classlabel);

train_data_mean=mean(mean(train_data,2));
sigma=std2(train_data);
train_data=(train_data-train_data_mean)./sigma;
test_data=(test_data-train_data_mean)./sigma;

iteration = 1000;%iteration
learning_rate_0 = 0.1;%initial learning rate
learning_rate = learning_rate_0;
effective_width_0 = 1;
effective_width = effective_width_0;
time_constant = iteration/log(effective_width_0);
w = rand(100,784);%randomly initialise all weights
[I,J] = ind2sub([10,10],1:100);%the positions of neurons in the som
no_neuron = 100;

%%
%Caculation
for n=1:iteration
    for i=1:600
        [~,winIdx] = min(dist(train_data(:,i)',w'));
        [winrow,wincolumn] = ind2sub([10,10],winIdx);
		win = [winrow,wincolumn];
		d = exp(-sum(([I(:) J(:)] - repmat(win,100,1)).^2,2)/(2*effective_width^2));        
        for j=1:no_neuron
            w(j,:) = w(j,:) + learning_rate*d(j).*(train_data(:,i)' - w(j,:));
        end
    end
    learning_rate = learning_rate_0*exp(-n/iteration);
    effective_width = effective_width_0*exp(-n/time_constant);
end
%%
%Output
for i = 1:100
        subplot(10,10,i);
        imshow(reshape(w(i,:),[28 28]));
end
save('result');
