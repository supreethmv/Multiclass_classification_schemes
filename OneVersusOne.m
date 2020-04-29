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
bias = zeros ( numClasses );
alpha = cell(numClasses,numClasses);
accumulator = zeros(length(Ytrain),numClasses,numClasses);

tic
for jj=1: numClasses
    
    for kk=jj+1: numClasses
        jkIndex = find(Ytrain == u(jj)==1 | Ytrain == u(kk)==1);% Group Selection, [True=1, false=0]
        ClassLabels = 2*(Ytrain(jkIndex )== u(jj))-1; % mapping [True=1, false=-1]
        [alpha{jj,kk},bias(jj,kk)]= getKernelSVMSolution(KernallTrain(jkIndex ,jkIndex ) ,ClassLabels ,C);
        accumulator(:,jj,kk) = KernallTrain(:,jkIndex) *( alpha {jj,kk }.* ClassLabels )+bias(jj,kk) >0;
        accumulator(:,kk,jj) = 1-accumulator (:,jj,kk) ;
    end
    
end

[~ , PredTrain ] = max(sum(accumulator,3),[],2);PredTrain=PredTrain-1;

toc
% Peak identification
disp(['Training Error:' num2str(sum(PredTrain~=Ytrain))])

%%%%%%%%%%%%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accumulator = zeros(length(Ytest),numClasses,numClasses);
tic
for jj=1: numClasses
    
    for kk=jj+1: numClasses
        jkIndex = find(Ytrain == u(jj)==1 | Ytrain == u(kk)==1);% Group Selection, [True=1, false=0]
        ClassLabels = 2*(Ytrain(jkIndex )== u(jj))-1; % mapping [True=1, false=-1]
        accumulator(:,jj,kk) = KernallTest(:,jkIndex) *( alpha {jj,kk }.* ClassLabels )+bias(jj,kk) >0;
        accumulator(:,kk,jj) = 1-accumulator (:,jj,kk) ;
    end
    
end
% Peak identification
[~ , PredOneVersusOne ] = max(sum(accumulator,3),[],2);PredOneVersusOne=PredOneVersusOne-1;
toc
% Error
[missmatchIdx,v] = find(PredOneVersusOne~=Ytest);
disp(['Test Error:' num2str(sum(v))])


% Save results
save USPSResultsOne.mat PredOneVersusOne


% Plotting
handles=VecToImage(Xtest(missmatchIdx,:),16,16,0,2,1);
for ii=1:length(missmatchIdx)
    
    subplot(handles(ii));
    title(['True:',num2str(Ytest(missmatchIdx(ii))),...
        ' Pred:',num2str(PredOneVersusOne(missmatchIdx(ii)))]);
    axis image
end

keyboard