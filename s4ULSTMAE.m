% Data set 1: 35mer Adenosine
% Anomaly Detection - Autoencoder ULSTM based


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

% Training AE- BLSTM network
numInputFeature = size(SegTrain(1),1)

% rng("default")
featureDimension = 1;

% Define biLSTM network layers
layers = [ sequenceInputLayer(featureDimension, 'Name', 'in','Normalization','none')
    lstmLayer(32, 'Name', 'bilstm1')
    reluLayer('Name', 'relu1')
    lstmLayer(16, 'Name', 'bilstm2')
    reluLayer('Name', 'relu2')
    lstmLayer(32, 'Name', 'bilstm3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(featureDimension, 'Name', 'fc')
    regressionLayer('Name', 'out') ];

% Set Training Options
options = trainingOptions('adam', ...
    'Plots', 'none', ...
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
SPE = vecnorm((ytrueNum'-ypredNum')).^2;


% MSE KDE
alpha = 0.1;

Hopt = 1.06* std(mse)*( length(mse)^ (-0.2));
pd_kernelmse = fitdist(mse','Kernel','Kernel','normal','Width',Hopt);
CIKdemse = icdf(pd_kernelmse,[alpha/2,1-alpha/2])

indNormalmse = (mse>=CIKdemse(1)) & (mse<=CIKdemse(2));
numNormalmseKDE = nnz(indNormalmse);

BiLabelTrainPredIndMSE = indNormalmse' ;
segIdNormalTrain = segIdTrain(BiLabelTrainPredIndMSE);
numNormalTrain= nnz(BiLabelTrainPredIndMSE);

valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
BiLabelTrainPredMSE = categorical(BiLabelTrainPredIndMSE,valueSet, catNames);

valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
BiLabelTrainPredMSE = categorical(BiLabelTrainPredIndMSE,valueSet, catNames);

xvalue = -0.05:0.001:0.15;
mseTrain = pd_kernelmse;
mseTrainpdf= pdf(mseTrain,xvalue);


figure;
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
plot(xvalue,mseTrainpdf,"LineWidth",1,"DisplayName","Density Function")
hold on
histogram(mse,20,'FaceColor',"#A2142F","DisplayName","Training Set")
xline([CIKdemse(1) CIKdemse(2)],'-- r', {'Lower Bound','Upper Bound'},"HandleVisibility","off","LineWidth",1)
xlabel("Prediction Error")
ylabel("Frequency")
xticklabels(strrep(xticklabels,'-','$-$'));
xlim([-0.05 0.15])
legend
hold off

figName= "AdePredErrorTrainULSTM.jpg";
figPath = "Data\Image\";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',1200);

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





numNormalMSEKDETest = nnz(indNormalLSTMTest)

TestPredIndLSTM = indNormalLSTMTest' ;
numNormalTestMSE = nnz(TestPredIndLSTM);
SegIdNormalTestMSE = segIdTest(TestPredIndLSTM);

% Make a new column for label Prediction MSE
valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
TestPredLSTM = categorical(TestPredIndLSTM,valueSet, catNames);


xline([CIKdemse(1) CIKdemse(2)],'-- r', {'Lower Bound','Upper Bound'},...
    "HandleVisibility","off","LineWidth",1)
hold on
xlabel("Prediction Error")
ylabel("Frequency")
plot(xvalue,mseTestpdf,"LineWidth",1,"DisplayName","Density Function")
histogram(mseTest,500,"FaceColor","#7E2F8E","DisplayName","Test Set")
legend
xticklabels(strrep(xticklabels,'-','$-$'));
xlim([-0.05, 0.15])
hold off

figName= "AdePredErrorTestULSTM.jpg";
figPath = "Data\Image\";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',1200);


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

TestPredFinal = TestPredLSTM;
confusionchart(TestAct,TestPredFinal)
% confusionchart(TestAct,TestPredFinal,'Normalization','total-normalized')

figName= "confAdeUlstm.jpg";
figPath = imagePath;
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',600);


% Removing the Test segments considered as an anomaly 
TestAnomalyPredSeg = segIdTest(~indNormalLSTMTest);

% Adding Reconstructed Seg to sTest 
s = sTest;
AC = ["0M","1nM","10nM","100nM","1uM","10uM"];
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

 %Structure - Anomaly LSTM
s = sTestLSTM;
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(TestAnomalyPredSeg, ACLoop);
    segLoop = TestAnomalyPredSeg(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoop = extractBefore(segLoop,patternLoop)


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

% Structure - Normal LSTM
% Selecting the Test segments considered as an normal 
TestNormalPredSeg = segIdTest(indNormalLSTMTest)

s = sTestLSTM;
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(TestNormalPredSeg, ACLoop);
    segLoop = TestNormalPredSeg(indLoop);
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

sTestNormalLSTM = s;
clear s;



TestPredFinalDouble = AnomalyTestTable.LSTMLogic;
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
textName= 'ULSTM';
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
% sTestAnomalyLSTM- Anomaly LSTM structure
structAnomalyLSTMName = strcat(textName,'-','TestsAnomalyLSTM');
structAnomalyLSTMFileName = strcat(restPath,structAnomalyLSTMName);
save(structAnomalyLSTMFileName,'sTestAnomalyLSTM');
% sTestNormalLSTM- Noraml LSTM structure
structNormalLSTMName = strcat(textName,'-','TestsNormalLSTM');
structNormalLSTMFileName = strcat(restPath,structNormalLSTMName);
save(structNormalLSTMFileName,'sTestNormalLSTM');
% Workspace variables
workspaceName= strcat(textName,'-','WorkSpaceVars');
workspaceFileName = strcat(restPath,workspaceName);
save(workspaceFileName)






