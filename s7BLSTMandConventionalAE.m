% Data set 1: 35mer Adenosine
% BLSTM & Plain AE

close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
% Load Segments
filePathTrain = "DataTrain\dataTrain.mat";
load(filePathTrain);
sTrain = s;
clear s;

filePathTest = "DataTest\dataTest.mat";
load(filePathTest);
sTest = s;
clear s;


tablePath = "Result\Table\";
figPath = "Result\File\";
restPath ="Result\Image\";

% Making array Seg of all segments
% Train Data Preparation
SegTrain = []; 
SegTitleTrain = [];
BiLabelTrainAct = [];
signalLabelsTrainActual = [];
ConcenTrain= [];
segIdTrain = [];


for i=1:size(sTrain,2)
    loopTable = sTrain(i).zscore;
    loopTitle = loopTable.Properties.VariableNames;
    loopTitle = loopTitle';
    SegTitleTrain = [SegTitleTrain;loopTitle];

    loopConcen = sTrain(i).AC;
    loopConcen = repmat(loopConcen, size(loopTable, 2),1);
    ConcenTrain = [ConcenTrain; loopConcen];

    loopSegId = strcat(loopTitle,'-',loopConcen);
    segIdTrain =  [segIdTrain; loopSegId];
    
    loopBiLabel = table2cell(sTrain(i).BiSegLabel)';
    BiLabelTrainAct = [BiLabelTrainAct; loopBiLabel];
    
    loopSignalLabel = table2cell(sTrain(i).SegLabel)';
    signalLabelsTrainActual = [signalLabelsTrainActual;loopSignalLabel];

    for j= 1:size(loopTable,2)
        currentLoop = loopTable.(j);
        currentLoop = currentLoop';
        SegTrain= [SegTrain;currentLoop];     
    end
end

SegTrainT = num2cell(SegTrain,2);
SegTrainSeq = SegTrainT';

BiLabelTrainAct = categorical(BiLabelTrainAct);

% Test Data Preparation
SegTest = []; 
SegTitleTest = [];
TestAct = [];
signalLabelsTestActual = [];
ConcenTest= [];
segIdTest = [];

for i=1:size(sTest,2)
    loopTable = sTest(i).zscore;
    loopTitle = loopTable.Properties.VariableNames;
    loopTitle = loopTitle';
    SegTitleTest = [SegTitleTest;loopTitle];
    
    loopConcen = sTest(i).AC;
    loopConcen = repmat(loopConcen, size(loopTable, 2),1);
    ConcenTest = [ConcenTest; loopConcen];
    
    loopSegId = strcat(loopTitle,'-',loopConcen);
    segIdTest =  [segIdTest; loopSegId];

    loopBiLabel = table2cell(sTest(i).BiSegLabel)';
    TestAct = [TestAct; loopBiLabel];
    
    loopSignalLabel = table2cell(sTest(i).SegLabel)';
    signalLabelsTestActual = [signalLabelsTestActual;loopSignalLabel];

    for j= 1:size(loopTable,2)
        currentLoop = loopTable.(j);
        currentLoop = currentLoop';
        SegTest= [SegTest;currentLoop];     
    end
end

SegTestT = num2cell(SegTest,2);
SegTestSeq = SegTestT';
TestAct = categorical(TestAct);

% LSTM
% Training AE- BLSTM network
numInputFeature = size(SegTrain(1),1);


featureDimension = 1;

% Define biLSTM network layers
layers = [ sequenceInputLayer(featureDimension, 'Name', 'in','Normalization','none')
    bilstmLayer(32, 'Name', 'bilstm1')
    reluLayer('Name', 'relu1')
    bilstmLayer(16, 'Name', 'bilstm2')
    reluLayer('Name', 'relu2')
    bilstmLayer(32, 'Name', 'bilstm3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(featureDimension, 'Name', 'fc')
    regressionLayer('Name', 'out') ];

% Set Training Options
options = trainingOptions('adam', ...
    'Plots', 'training-progress', ...
    'MiniBatchSize', 5,...
    'MaxEpochs',100,...
    'Plots','none','Verbose',false);

autoenc = trainNetwork(SegTrainSeq, SegTrainSeq, layers, options);

% Validation Section
SegTrainReconstructed = predict(autoenc, SegTrainSeq);

ytrueTrain = SegTrainSeq';
ypredTrain = SegTrainReconstructed;
ytrueNum =  cell2mat(ytrueTrain);
ypredNum = cell2mat(ypredTrain);


mse = mean((ytrueNum'-ypredNum').^2);
SPE = vecnorm((ytrueNum'-ypredNum')).^2


% MSE KDE
alpha = 0.1;

Hopt = 1.06* std(mse)*( length(mse)^ (-0.2));
pd_kernelmse = fitdist(mse','Kernel','Kernel','normal','Width',Hopt);
CIKdemse = icdf(pd_kernelmse,[alpha/2,1-alpha/2]);

indNormalmse = (mse>=CIKdemse(1)) & (mse<=CIKdemse(2));
numNormalmseKDE = nnz(indNormalmse);

BiLabelTrainPredIndMSE = indNormalmse' ;
segIdNormalTrainlstm = segIdTrain(BiLabelTrainPredIndMSE);
numNormalTrain= nnz(BiLabelTrainPredIndMSE);

valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
BiLabelTrainPredMSE = categorical(BiLabelTrainPredIndMSE,valueSet, catNames);


% Change segTrainReconstructed from Cell to the mat
segReconstLstmT = cell2mat(SegTrainReconstructed);
segReconstLstm = segReconstLstmT';

s = sTrain;
AC = ["0M","1nM","10nM","100nM","1uM","10uM"];

for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(segIdTrain, ACLoop);
    segLoop = segReconstLstm(:, indLoop);
    
    
    patternLoop = strcat('-',ACLoop);
    varLoop = extractBefore(segIdTrain(indLoop),patternLoop);
    
    segLoopTbl = array2table(segLoop);
    segLoopTbl.Properties.VariableNames = varLoop;
    
    s(i).ReconLstm = segLoopTbl;
  
end

sTrain = s;
clear s;

% Test Section
SegTestReconstructed = predict(autoenc,SegTestSeq);

ytrueTest = SegTestSeq';
ypredTest = SegTestReconstructed;


ytrueNumTest =  cell2mat(ytrueTest);
ypredNumTest = cell2mat(ypredTest);


 % MSE KDE Test

mseTest = mean((ytrueNumTest'-ypredNumTest').^2);

indNormalLSTMTest = (mseTest>=CIKdemse(1)) & (mseTest<=CIKdemse(2));

segTestAnomalyLSTMNum = find(indNormalLSTMTest' == 0);
% the following index should be replaced with Plain AE results (segTestAnomalyPlainAENum)
% It shows the index of Non Identified segments in the Anomaly Table Test
% the following index should be replaced with Plain AE results


segTestPlainAENum = find(indNormalLSTMTest' == 1);
segTestPlainAE = SegTest(segTestPlainAENum,:);

numNormalMSEKDETest = nnz(indNormalLSTMTest);

TestPredIndLSTM = indNormalLSTMTest' ;
numNormalTestMSE = nnz(TestPredIndLSTM);
SegIdNormalTestMSE = segIdTest(TestPredIndLSTM);

% Make a new column for label Prediction MSE
valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
TestPredLSTM = categorical(TestPredIndLSTM,valueSet, catNames);


valueSet = 0:1;
catNames = {'Anomaly' 'N/I' };
TestPredLSTMTransfer = categorical(TestPredIndLSTM,valueSet, catNames);
logicLSTMTest =   indNormalLSTMTest' ;



% Table - Test - Anomaly
AnomalyTestTable = table();

AnomalyTestTable.segIdTest = segIdTest;
AnomalyTestTable.ActTest = TestAct;

AnomalyTestTable.PredTestLSTM = TestPredLSTM;
AnomalyTestTable.LSTMLogic =  logicLSTMTest;

AnomalyTestTable.PredTestLSTMTransfer = TestPredLSTMTransfer;



% confusionchart(TestAct,TestPredLSTM)
% 
% confusionchart(TestAct,TestPredLSTM,'Normalization','total-normalized')
% confusionchart(TestAct,TestPredLSTM,'Normalization','row-normalized')
% confusionchart(TestAct,TestPredLSTM,'Normalization','column-normalized')

%Removing the Test segments considered as an anomaly 
TestAnomalyPredSeg = segIdTest(~indNormalLSTMTest);

% Making new Test structure
s = sTest;
AC = ["0M","1nM","10nM","100nM","1uM","10uM"];

for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(TestAnomalyPredSeg, ACLoop);
    segLoop = TestAnomalyPredSeg(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoop = extractBefore(segLoop,patternLoop);


    s(i).DrainNA = removevars(s(i).DrainNA,varLoop);
    s(i).zscore = removevars(s(i).zscore,varLoop);
    s(i).SignalLabel = removevars(s(i).SignalLabel,varLoop);
    s(i).SegLabel = removevars(s(i).SegLabel,varLoop);
    s(i).BiSegLabel = removevars(s(i).BiSegLabel,varLoop);   
end

sTestAE = s;
clear s;

%Adding Reconstructed Seg to sTest 
s = sTest;

for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(segIdTest, ACLoop);
    segIdLoop = segIdTest(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoopT = extractBefore(segIdLoop,patternLoop);
    varLoop = varLoopT';
 

    segLoop = SegTestReconstructed(indLoop);
    segLoopMatT= cell2mat(segLoop);
    segLoopMat = segLoopMatT';
    segLoopTbl = array2table(segLoopMat);
    segLoopTbl.Properties.VariableNames = varLoop;
    s(i).ReconSeg = segLoopTbl; 
end

sTestLSTM = s;
clear s;

% Making structure of Anomaly Structure detected by LSTM
s = sTestLSTM;
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(TestAnomalyPredSeg, ACLoop);
    segLoop = TestAnomalyPredSeg(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoop = extractBefore(segLoop,patternLoop);


    s(i).DrainNA = s(i).DrainNA(:,varLoop);
    s(i).zscore =s(i).zscore(:,varLoop);
    s(i).zscore = s(i).zscore(:,varLoop);
    s(i).SignalLabel = s(i).SignalLabel(:,varLoop);
    s(i).SegLabel = s(i).SegLabel(:,varLoop);
    s(i).BiSegLabel = s(i).BiSegLabel(:,varLoop);
    s(i).ReconSeg = s(i).ReconSeg(:,varLoop);
end

sTestAnomalyLSTM = s;
clear s;

% Plain AE
% Training AE
SegTrain = SegTrain';

autoencPlain = trainAutoencoder(SegTrain,16);

SegTrainReconstPlainAE = predict(autoencPlain, SegTrain);
ytrueTrain = SegTrain;
ypredTrain = SegTrainReconstPlainAE;

msePlainAE = mean((ytrueTrain-ypredTrain).^2);


% Train AE - MSE KDE CI
alpha = 0.1;

HoptPlainAE = 1.06* std(msePlainAE)*( length(msePlainAE)^ (-0.2));
pd_kernelAE = fitdist(msePlainAE','Kernel','Kernel','normal','Width',HoptPlainAE);
CIKdeAE = icdf(pd_kernelAE,[alpha/2,1-alpha/2]);


indNormalPlainAE = (msePlainAE>=CIKdeAE(1)) & (msePlainAE<=CIKdeAE(2));
numNormalmseAE = nnz(indNormalPlainAE);
segIdTrainNormalPAE = segIdTrain(indNormalPlainAE);

s = sTrain;
AC = ["0M","1nM","10nM","100nM","1uM","10uM"];

for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(segIdTrain, ACLoop);
    segLoop = SegTrainReconstPlainAE(:, indLoop);
    
    
    patternLoop = strcat('-',ACLoop);
    varLoop = extractBefore(segIdTrain(indLoop),patternLoop);
    
    segLoopTbl = array2table(segLoop);
    segLoopTbl.Properties.VariableNames = varLoop;
    
    s(i).ReconPAE = segLoopTbl;
  
end

sTrain = s;
clear s;

% Test AE
segTestReconstAE = predict(autoencPlain,segTestPlainAE');
ytrueTest = segTestPlainAE';
ypredTest = segTestReconstAE;

mseTest = mean((ytrueTest-ypredTest).^2);
indNormalPlainAETest = (mseTest>=CIKdeAE(1)) & (mseTest<=CIKdeAE(2));
numNormalmseAE = nnz(indNormalPlainAETest);


TestPredIndPlainAE = indNormalPlainAETest' ;
numNormalTestPlainAE = nnz(TestPredIndPlainAE);
SegIdNormalTestPlainAE = SegIdNormalTestMSE(TestPredIndPlainAE);

valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
TestPredPlainAE = categorical(TestPredIndPlainAE,valueSet, catNames);



TestPredAE= repmat("N/I",size(TestPredLSTM,1),1);
TestPredAE(segTestPlainAENum) = TestPredPlainAE;

logicPlainAETest= logicLSTMTest;
logicPlainAETest(segTestPlainAENum) = TestPredIndPlainAE;


% Anomaly Test Table Modification with Plain AE 
AnomalyTestTable.PredPlainAE = TestPredAE;
AnomalyTestTable.PlainAE = logicPlainAETest;

% Adding Reconstructed Seg to sTestAE
s = sTestAE;
 
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(SegIdNormalTestMSE, ACLoop);
    segIdLoop = SegIdNormalTestMSE(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoopT = extractBefore(segIdLoop,patternLoop);
    varLoop = varLoopT';
 

    segLoop = segTestReconstAE(:,indLoop);
    segLoopTbl = array2table(segLoop);
    segLoopTbl.Properties.VariableNames = varLoop;
    s(i).ReconSeg = segLoopTbl; 
end
 
sTestAERecon = s;
clear s;

% Test -  Anomaly AE

% Selecting the Test segments considered as an anomaly 
TestAnomalyAE = SegIdNormalTestMSE(~TestPredIndPlainAE);

% Making structure of Anomaly Segments detected by AE
s = sTestAERecon;
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(TestAnomalyAE, ACLoop);
    segIdLoop = TestAnomalyAE(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoopT = extractBefore(segIdLoop,patternLoop);
    varLoop = varLoopT';
 
    s(i).DrainNA = s(i).DrainNA(:,varLoop);
    s(i).zscore =s(i).zscore(:,varLoop);
    s(i).zscore = s(i).zscore(:,varLoop);
    s(i).SignalLabel = s(i).SignalLabel(:,varLoop);
    s(i).SegLabel = s(i).SegLabel(:,varLoop);
    s(i).BiSegLabel = s(i).BiSegLabel(:,varLoop);
    s(i).ReconSeg = s(i).ReconSeg(:,varLoop);
end
 
sTestAnomalyAE = s;
clear s;

% Test - Normal AE

% Selecting the Test segments considered as an anomaly 
TestNormalAE = SegIdNormalTestMSE(TestPredIndPlainAE);
% Making structure of Normal Segments detected by AE
s = sTestAERecon;
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(TestNormalAE, ACLoop);
    segIdLoop = TestNormalAE(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoopT = extractBefore(segIdLoop,patternLoop);
    varLoop = varLoopT';
 
    s(i).DrainNA = s(i).DrainNA(:,varLoop);
    s(i).zscore =s(i).zscore(:,varLoop);
    s(i).zscore = s(i).zscore(:,varLoop);
    s(i).SignalLabel = s(i).SignalLabel(:,varLoop);
    s(i).SegLabel = s(i).SegLabel(:,varLoop);
    s(i).BiSegLabel = s(i).BiSegLabel(:,varLoop);
    s(i).ReconSeg = s(i).ReconSeg(:,varLoop);
end
 
sTestNormalAE = s;
clear s;

% Prediction final
TestPredFinal = TestPredLSTM;
TestPredFinal(segTestPlainAENum) = TestPredPlainAE;


AnomalyTestTable.PredFinal = TestPredFinal;
AnomalyTestTable.Final = logicPlainAETest;


% Confusion Chart Final
confusionchart(TestAct,TestPredFinal)
% confusionchart(TestAct,TestPredFinal,'Normalization','total-normalized')

figName= "confAdeBlstmPAE.jpg";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',600);



TestPredFinalDouble = AnomalyTestTable.Final;
TestActualDouble = AnomalyTestTable.ActTest =="Good";

[c_matrix,Result]= confusion.getMatrix(TestActualDouble,TestPredFinalDouble);

metricTable = table;
metricTable(1,:) =cell2table({"Accuracy" ,"Sensitivity", ...
    "Specificity", "Precision", "F1score"});

acc = Result.Accuracy;
sens = Result.Sensitivity;
spec = Result.Specificity;
prec = Result.Precision;
f1score = Result.F1_score;

metricArray = [acc sens spec prec f1score];
metricCell = num2cell(metricArray);

metricTable(2,:) = metricCell;


% Save 
textName= 'BLSTMandConventionalAE';

% sTrain
structTrainName = strcat(textName,'-','TrainLstmAndPAE');
structTrainFileName = strcat(restPath,structTrainName);
save(structTrainFileName,'sTrain');
% AnomalyTestTable
TestTableFileName= strcat(restPath, textName,'-TestAnomalyTable.csv');
writetable(AnomalyTestTable, TestTableFileName);
% MetricTable

metricTableFileName= strcat(restPath, textName,'-MetricTable.csv');
writetable( metricTable,metricTableFileName);

% sTestLSTM structure - Reconstructed
structLSTMName = strcat(textName,'-','TestsLSTM');
structLSTMFileName = strcat(restPath,structLSTMName);
save(structLSTMFileName,'sTestLSTM');
% Anomaly LSTM structure
structAnomalyLSTMName = strcat(textName,'-','TestsAnomalyLSTM');
structAnomalyLSTMFileName = strcat(restPath,structAnomalyLSTMName);
save(structAnomalyLSTMFileName,'sTestAnomalyLSTM');
% sTestAE structure - Reconstructed
structAEName = strcat(textName,'-','TestsAE');
structAEFileName = strcat(restPath,structAEName);
save(structAEFileName,'sTestAERecon');
% sTestAnomalyAE
structAnomalyAEName = strcat(textName,'-','TestsAnomalyAE');
structAnomalyAEFileName = strcat(restPath,structAnomalyAEName);
save(structAnomalyAEFileName,'sTestAnomalyAE');
% sTestNormalAE
structNormalAEName = strcat(textName,'-','TestsNormalAE');
structNormalAEFileName = strcat(restPath,structNormalAEName);
save(structNormalAEFileName,'sTestNormalAE');
%  workspace variables
workspaceName= strcat(textName,'-','WorkSpaceVars');
workspaceFileName = strcat(restPath,workspaceName);
save(workspaceFileName)