clc; clear all; close all;

%% reading data 
data = readtable('features.csv');
data = table2array(data);
labels = readtable('output.csv');
labels = table2array(labels);
labels = labels(:,2);
genald = data(:,1); 


patt = zeros(15485,5);


count = 1;
for i= 1:100:length(genald)
    ii = i + 99;
    data_updated = data(i:ii,2:6);
    [evectors, score, evalues] = pca(data_updated'); % Calculate eigenvectors and eigenvalues
    component = score(:,1);
    %feature = feature';
    patt(count,:) = component(:,1);
    count = count + 1;
end

x1 = patt(:,1);
x2 = patt(:,2);
x3 = patt(:,3);
x4 = patt(:,4);
x5 = patt(:,5);



%%dividing testing and training data
tmpVec = randperm(size(labels,1));
M = 10840;
x_train = [x1(tmpVec(1:M)), x2(tmpVec(1:M)), x3(tmpVec(1:M)), x4(tmpVec(1:M)), x5(tmpVec(1:M))];
x_train = abs(x_train);
y_train = labels(tmpVec(1:M));
x_test = [x1(tmpVec(M+1:end)), x2(tmpVec(M+1:end)), x3(tmpVec(M+1:end)), x4(tmpVec(M+1:end)), x5(tmpVec(M+1:end))];
y_test = labels(tmpVec(M+1:end));
y_train = categorical(y_train);



%% Logistic regression

beta = mnrfit(x_train, y_train);
y_logit_prob = mnrval(beta, x_test);
[~, y_logit_pred] = max(y_logit_prob');


%% Post-processing logistic regression model
cp = classperf(y_test);
cp = classperf(cp, y_logit_pred);
modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Model accuracy = %0.3f\n', modelAccuracy); 
modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Model sensitivity = %0.3f\n', modelSensitivity);
modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Model specificity = %0.3f\n', modelSpecificity);

%% Estimating area under curve
[X, Y, ~, AUC] = perfcurve(y_test, y_logit_prob(:,1), 1); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

%% Plotting the ROC curve 
figure; plot(X, Y,'b-','LineWidth',2); 
title('ROC curve for logistic regression','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);



%%Feature engineering using two new feature vectors

for i = 1:length(x1)  % used a for loop to find create a new feature by using the mean of two old features
    x6(i) = mean(x1(i));
end 
for i = 1:length(x2)
    x7(i) = mean(x2(i));
end 

 x6 = x6';
 x7 = x7';

 data2_updated = [patt, x6,x7]; %update the old matrix with the two new engineered features
 l = 1;
 
for a= 1:length(x3)
    
    data3_updated = data2_updated(a,1:7);
    [evectors, score, evalues] = pca(data3_updated');
    component = score(:,1);
    patt_updated(l,:) = component(:,1);
    l = l + 1;
end

%creates a new pattern matrix by updating the old one matrix

x1n = patt_updated(:,1);
x2n = patt_updated(:,2);
x3n = patt_updated(:,3);
x4n = patt_updated(:,4);
x5n = patt_updated(:,5);
x6n = patt_updated(:,6);
x7n = patt_updated(:,7);



%%dividing testing and training data new
tmpVec_1 = randperm(size(labels,1));
M = 10840;
x_train_1 = [x1n(tmpVec_1(1:M)), x2n(tmpVec_1(1:M)), x3n(tmpVec_1(1:M)), x4n(tmpVec_1(1:M)), x5n(tmpVec_1(1:M)),x6n(tmpVec_1(1:M)),x7n(tmpVec_1(1:M))];
x_train_1 = abs(x_train_1);
y_train_1 = labels(tmpVec_1(1:M));
y_train_1 = categorical(y_train_1);
x_test_1 = [x1n(tmpVec_1(M+1:end)), x2n(tmpVec_1(M+1:end)), x3n(tmpVec_1(M+1:end)), x4n(tmpVec_1(M+1:end)), x5n(tmpVec_1(M+1:end)),x6n(tmpVec_1(M+1:end)),x7n(tmpVec_1(M+1:end))];
y_test_1 = labels(tmpVec_1(M+1:end));

%% Logistic regression new

beta_1 = mnrfit(x_train_1, y_train_1);
y_logit_prob_1 = mnrval(beta_1, x_test_1);
[~, y_logit_pred_1] = max(y_logit_prob_1');

%% Post-processing logistic regression model new
cp1 = classperf(y_test_1);
cp1 = classperf(cp1, y_logit_pred_1);
modelAccuracy1 = cp1.CorrectRate; % Model accuracy 
fprintf('Model accuracy1 = %0.3f\n', modelAccuracy1); 
modelSensitivity1 = cp1.Sensitivity; % Model sensitivity 
fprintf('Model sensitivity1 = %0.3f\n', modelSensitivity1);
modelSpecificity1 = cp1.Specificity; % Model specificity 
fprintf('Model specificity1 = %0.3f\n', modelSpecificity1);

%% Estimating area under curve new
[X1, Y1, ~, AUC1] = perfcurve(y_test_1, y_logit_prob_1(:,1), 1); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC1); 

%% Plotting the ROC curve new
figure; plot(X1, Y1,'b-','LineWidth',2); 
title('ROC curve for logistic regression 1','FontSize',14,'FontWeight','bold');
xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 
set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);


