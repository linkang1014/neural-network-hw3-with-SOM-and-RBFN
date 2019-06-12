%%
%Input
clc
clear all;
close all;
x = linspace(-pi,pi,400);
trainX = [x; sinc(x)]; % 2x400 matrix
Index = randperm(size(trainX,1));
trainX_random = trainX(Index,:);
% plot(trainX(1,:),trainX(2,:),'+r');% axis equal

iteration = 500;%iteration
learning_rate_0 = 0.1;%initial learning rate
learning_rate = learning_rate_0;
effective_width_0 = 1;
effective_width = effective_width_0;
time_constant = iteration/log(effective_width_0);

no_neuron = 40;
w = rand(40,2);%randomly initialise all weights
[I,J] = ind2sub([1,no_neuron],1:40);%the positions of neurons in the som

%%
%Caculation
for n=1:iteration
    for i=1:400
        [~,winIdx] = min(dist(trainX_random(:,i)',w'));
        [winrow,wincolumn] = ind2sub([1,40],winIdx);
		win = [winrow,wincolumn];
		d = exp(-sum(([I(:) J(:)] - repmat(win,40,1)).^2,2)/(2*effective_width^2));        
        for j=1:no_neuron
            w(j,:) = w(j,:) + learning_rate*d(j).*(trainX_random(:,i)' - w(j,:));
        end
    end
    learning_rate = learning_rate_0*exp(-n/iteration);
    effective_width = effective_width_0*exp(-n/time_constant);
end
%%
%Output
figure
plot(trainX(1,:),trainX(2,:),'*r',w(:,1),w(:,2),'o-bl');
legend('train data','som result');
set(gca,'FontSize',12);
xlabel('x','FontSize',16);
ylabel('sinc(x)','FontSize',16);
title('result of Q3(a)','FontSize',20);