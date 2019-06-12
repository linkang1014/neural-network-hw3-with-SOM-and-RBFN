%%
%Input
clc
clear all;
close all;
X = randn(800,2);
s2 = sum(X.^2,2);
trainX = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';

iteration = 500;%iteration
learning_rate_0 = 0.1;%initial learning rate
learning_rate = learning_rate_0;
effective_width_0 = 1;
effective_width = effective_width_0;
time_constant = iteration/log(effective_width_0);

no_neuron = 64;
w = rand(64,2);%randomly initialise all weights
[I,J] = ind2sub([8,8],1:64);%the positions of neurons in the som

%%
%Caculation
for n=1:iteration
    for i=1:400
        [~,winIdx] = min(dist(trainX(:,i)',w'));
        [winrow,wincolumn] = ind2sub([8,8],winIdx);
		win = [winrow,wincolumn];
		d = exp(-sum(([I(:) J(:)] - repmat(win,64,1)).^2,2)/(2*effective_width^2));        
        for j=1:no_neuron
            w(j,:) = w(j,:) + learning_rate*d(j).*(trainX(:,i)' - w(j,:));
        end
    end
    learning_rate = learning_rate_0*exp(-n/iteration);
    effective_width = effective_width_0*exp(-n/time_constant);
end
%%
%Output
figure
plot(trainX(1,:),trainX(2,:),'+r',w(:,1),w(:,2),'obl');
axis equal;
hold on;
for i = 0:7
    plot(w(i*8+1:(i+1)*8,1),w(i*8+1:(i+1)*8,2),'-dk');
end
hold on;
for i = 1:8
    plot(w(i:8:i+56,1),w(i:8:i+56,2),'-dk');
end
hold off;
legend('train data','som result');
set(gca,'FontSize',12);
xlabel('X','FontSize',16);
ylabel('Y','FontSize',16);
title('result of Q3(b)','FontSize',20);