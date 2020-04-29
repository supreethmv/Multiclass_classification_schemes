clear all
close all
clc

addpath = 'C:\Users\Danish\Desktop\ML Assignment Danish\DataEx7\libsvm-3.14\libsvm-3.14\matlab';
load USPSTrain.mat
load USPSTest.mat

C = 100;

u=unique(Ytrain);
numClasses=length(u);


% Squared Eculidean Distance
DistTrain = dist_euclidean(Xtrain,Xtrain); % Training
DistTest = dist_euclidean(Xtest,Xtrain); % Testing


% Gaussian Kernel
lambda = 3;
KernallTrain = exp(-lambda*DistTrain/median(DistTrain(:))); % Training
KernallTest = exp(-lambda*DistTest/median(DistTrain(:))); % Testing


%%%%%%%%%%%%%%%%%%%%%%%%%%Training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter Initialization
alpha = zeros(length(Ytrain),numClasses);
OutputTrain = zeros(length(Ytrain),numClasses);
bias = zeros(numClasses,1);
tic
% Training
for cid=1:numClasses
    ClassLabels = Ytrain==u(cid); % Group Selection, [True=1, false=0]
    ClassLabels = 2*(ClassLabels)-1; % mapping [True=1, false=-1]
    [alpha(:,cid),bias(cid)] = getKernelSVMSolution(KernallTrain,ClassLabels,C);
    OutputTrain(:,cid) = KernallTrain*(alpha(:,cid).*ClassLabels) + bias(cid);
end

% Peak identification
[~,PredTrain] = max(OutputTrain,[],2);PredTrain=PredTrain-1;
toc
disp(['Training Error:' num2str(sum(PredTrain~=Ytrain))])

%%%%%%%%%%%%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OutputTest = zeros(length(Ytest),numClasses);
tic
% Testing
for cid=1:numClasses
    ClassLabels = Ytrain==u(cid); % Group Selection, [True=1, false=0]
    ClassLabels = 2*(ClassLabels)-1; % mapping [True=1, false=-1]
    OutputTest(:,cid) = KernallTest*(alpha(:,cid).*ClassLabels) + bias(cid);
end

% Peak identification
[~,PredOneVersusAll] = max(OutputTest,[],2);PredOneVersusAll=PredOneVersusAll-1;
toc
% Error
[missmatchIdx,v] = find(PredOneVersusAll~=Ytest);
disp(['Test Error:' num2str(sum(v))])


% Save results
save USPSResultsAll.mat PredOneVersusAll


% Plotting
handles=VecToImage(Xtest(missmatchIdx,:),16,16,0,2,1);
for ii=1:length(missmatchIdx)
    
    subplot(handles(ii)); 
    title(['True:',num2str(Ytest(missmatchIdx(ii))),...
        ' Pred:',num2str(PredOneVersusAll(missmatchIdx(ii)))]);
    axis image
end

keyboard