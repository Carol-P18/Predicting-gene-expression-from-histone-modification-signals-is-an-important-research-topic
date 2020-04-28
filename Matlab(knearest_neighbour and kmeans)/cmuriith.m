clc; clear all; close all;

%% Reading Data
data = readtable('features.csv');
data = table2array(data);

labels = readtable('output.csv');
labels = table2array(labels);

labels = labels(:,2);         %Extracting the output column
genald = data(:,1); %Extracting the Gene number column


patt = zeros(15485,5);


a = 1;
for l= 1:100:length(genald)
    ll = l + 99;
    data_updated = data(l:ll,2:6);
    [evectors evalues score] = pca(data_updated');
    component = evectors(:,1);
    component = component';
    patt(a,:) = component(:,1);
    a = a + 1;
end

x1 = patt(:,1);
x2 = patt(:,2);
x3 = patt(:,3);
x4 = patt(:,4);
x5 = patt(:,5);




%%
%%testing and dividing
M = 10840;
x_train = [x1(1:M), x2(1:M), x3(1:M), x4(1:M), x5(1:M)];
x_train = nnmf(x_train,2);
y_train = labels(1:M);
x_test1 = [x1(M+1:end), x2(M+1:end), x3(M+1:end), x4(M+1:end), x5(M+1:end)];
x_test1 = nnmf(x_test1,2);
y_test = labels(M+1:end);



%% Fit linear model (with linear terms)

lm1 = fitlm(x_train,y_train,'linear');
yPred1 = predict(lm1,x_test1);


%%%% Compute error

yErr1 = 0;
for ii=1:length(y_test)
    yErr1 = yErr1 + (yPred1(ii) - y_test(ii))^2;
end
yError1 = sqrt((yErr1)/length(y_test));
fprintf('Root mean square error using linear model (with linear terms) = %f\n', yError1);


%% Adding Feature engineering


for l=1:length(x1)
    x6(l)=mean(x1);
    x7(l)=mean(x2);
end

x6 = x6';
x7 = x7';
patt = [patt,x6,x7];
x1 = patt(:,1);
x2 = patt(:,2);
x3 = patt(:,3);
x4 = patt(:,4);
x5 = patt(:,5);
x6=  patt(:,6);
x7=  patt(:,7);

%%testing and dividing
M = 10840;
x_train1 = [x1(1:M), x2(1:M), x3(1:M), x4(1:M), x5(1:M),x6(1:M), x7(1:M)];
x_train1 = nnmf(x_train1,2);
y_train1 = labels(1:M);
x_test1 = [x1(M+1:end), x2(M+1:end), x3(M+1:end), x4(M+1:end), x5(M+1:end),x6(M+1:end), x7(M+1:end)];
x_test1 = nnmf(x_test1,2);
y_test = labels(M+1:end);



% Fit linear model (with linear terms)

lm22 = fitlm(x_train1,y_train1,'linear');
yPred2 = predict(lm22,x_test1);

%% Compute error

yErr2 = 0;
for ii=1:length(y_test)
    yErr2 = yErr2 + (yPred2(ii) - y_test(ii))^2;
end
yError2 = sqrt((yErr2)/length(y_test));
fprintf('Root mean square error using linear model (with linear terms) = %f\n', yError2);

%% Fit linear model (with only selected features)

X3 = x4;
x_train1 = X3(1:M,:);
y_train1 = labels(1:M);
x_test1 = X3(M+1:end,:);
y_test1 = labels(M+1:end);

%% Fit linear model (with linear terms)

lm3 = fitlm(x_train1,y_train1,'linear');
yPred3 = predict(lm3,x_test1);


%% Compute error

yErr3 = 0;
for ii=1:length(y_test1)
    yErr3 = yErr3 + (yPred3(ii) - y_test1(ii))^2;
end
yError3 = sqrt((yErr3)/length(y_test1));
fprintf('Root mean square error using linear model (with linear terms) = %f\n', yError3);

%% Variable importance

anova1 = anova(lm1);
anova2 = anova(lm22);
anova3 = anova(lm3);

%% Model with quadratic term

lm4 = fitlm(x_train1,y_train1,'quadratic');
yPred4 = predict(lm4,x_test1);

%% Compute error

yErr4 = 0;
for ii=1:length(y_test1)
    yErr4 = yErr4 + (yPred4(ii) - y_test1(ii))^2;
end
yError4 = sqrt((yErr4)/length(y_test1));
fprintf('Root mean square error using linear model (with linear terms) = %f\n', yError4);