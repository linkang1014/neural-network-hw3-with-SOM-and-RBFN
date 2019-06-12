%Input
clc
clear all;
close all;
load('result.mat');
%caculate the winner weight label
for k = 1:100
    for r = 1:600
        distance(r) = (w(k,:)-train_data(:,r)')*(w(k,:)-train_data(:,r)')';
    end
    point = find(distance==min(distance));
    win_index(k) = train_classlabel(point(1));
end

for k = 1:60
    for r = 1:100
        distance_test(r) = (w(r,:) - test_data(:,k)')*(w(r,:) - test_data(:,k)')';
    end
    point = find(distance_test==min(distance_test));
    test_index(k) = win_index(point(1));
end

%%
%Output
number = 0;
for i =1:60
    if test_index(i) == test_classlabel(i) 
        number = number+1;
    end
end
accuracy = number/60;
