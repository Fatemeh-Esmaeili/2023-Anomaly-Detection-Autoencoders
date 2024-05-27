% Data set 1: 35mer Adenosine
% Anomaly Detection - Autoencoder Conventional AE based


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
SegTrain = SegTrain';
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
SegTest = SegTest';
SegTestT = num2cell(SegTest,2);
SegTestSeq = SegTestT';
TestAct = categorical(TestAct);

% Training PAE
autoencPlain = trainAutoencoder(SegTrain,16);

% Validation Section
SegTrainReconstructed = predict(autoencPlain, SegTrain);

ytrueTrain = SegTrain;
ypredTrain = SegTrainReconstructed;


mse = mean((ytrueTrain-ypredTrain).^2);


s = sTrain;
AC = ["0M","1nM","10nM","100nM","1uM","10uM"];
for i = 1: size(s,2)
    ACLoop = AC(i);
    indLoop = contains(segIdTrain, ACLoop);
    segIdLoop = segIdTrain(indLoop);
    patternLoop = strcat('-',ACLoop);
    varLoopT = extractBefore(segIdLoop,patternLoop);
    varLoop = varLoopT';
 

    segLoop = SegTrainReconstructed(:, indLoop);    
    segLoopTbl = array2table(segLoop);
    segLoopTbl.Properties.VariableNames = varLoop;
    s(i).ReconSeg = segLoopTbl; 
end

sTrainPAE = s;
clear s;



% MSE KDE
alpha = 0.1;

Hopt = 1.06* std(mse)*( length(mse)^ (-0.2));
pd_kernelmse = fitdist(mse','Kernel','Kernel','normal','Width',Hopt);
CIKdemse = icdf(pd_kernelmse,[alpha/2,1-alpha/2]);

indNormalmse = (mse>=CIKdemse(1)) & (mse<=CIKdemse(2))

numNormalmseKDE = nnz(indNormalmse)
logicTrainSeg = indNormalmse'


xvalue = -0.05:0.001:0.08;
mseTrain = pd_kernelmse;
mseTrainpdf= pdf(mseTrain,xvalue);

figure;
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
plot(xvalue,mseTrainpdf,"LineWidth",1,"DisplayName","Density Function")
hold on
histogram(mse,20,'FaceColor',"#D95319","DisplayName","Training Set")
xline([CIKdemse(1) CIKdemse(2)],'-- r', {'Lower Bound','Upper Bound'},"HandleVisibility","off","LineWidth",1)
xlabel("Prediction Error")
ylabel("Frequency")
xticklabels(strrep(xticklabels,'-','$-$'));
xlim([-0.02 0.08])
legend
hold off

figName= "AdePredErrorTrainPAE.jpg";
figPath = "Data\Image\";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',1200);


% Test Section
SegTestReconstructed = predict(autoencPlain,SegTest);
ytrueTest = SegTest;
ypredTest = SegTestReconstructed;

% MSE KDE Test
mseTest = mean((ytrueTest-ypredTest).^2);
indNormalPAETest = (mseTest>=CIKdemse(1)) & (mseTest<=CIKdemse(2));
segTestAnomalyPAENum = find(indNormalPAETest' == 0);

numNormalMSEKDETest = nnz(indNormalPAETest);

TestPredIndPAE = indNormalPAETest' ;
numNormalTestMSE = nnz(TestPredIndPAE);
SegIdNormalTestMSE = segIdTest(TestPredIndPAE);


figure;
xvalue = -0.05:0.001:0.15;
mseTestpdf= pdf(mseTrain,xvalue);


xline([CIKdemse(1) CIKdemse(2)],'-- r', {'Lower Bound','Upper Bound'},...
    "HandleVisibility","off","LineWidth",1)
hold on
xlabel("Prediction Error")
ylabel("Frequency")
plot(xvalue,mseTestpdf,"LineWidth",1,"DisplayName","Density Function")
histogram(mseTest,1000,"FaceColor","#77AC30","DisplayName","Test Set")
legend
xticklabels(strrep(xticklabels,'-','$-$'));
xlim([-0.02 0.08])
hold off

figName= "AdePredErrorTestPAE.jpg";
figPath = "Data\Image\";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',1200);
% Make a new column for label Prediction MSE
valueSet = 0:1;
catNames = {'Anomaly' 'Good' };
TestPredPAE = categorical(TestPredIndPAE,valueSet, catNames);

logicPAETest =   indNormalPAETest' ;

 %Table - Test - Anomaly
AnomalyTestTable = table();

AnomalyTestTable.segIdTest = segIdTest;
AnomalyTestTable.ActTest = TestAct;

AnomalyTestTable.PredTestPAE = TestPredPAE;
AnomalyTestTable.PAELogic =  logicPAETest;

TestPredFinal = TestPredPAE;
fig = confusionchart(TestAct,TestPredFinal);
% fig = confusionchart(TestAct,TestPredFinal,'Normalization','total-normalized')


figName= "confAdePae.jpg";
figName = strcat(figPath,figName);
exportgraphics(gcf,figName,'Resolution',600);


% Removing the Test segments considered as an anomaly 
TestAnomalyPredSeg = segIdTest(~indNormalPAETest);

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
 

    segLoop = SegTestReconstructed(:, indLoop);    
    segLoopTbl = array2table(segLoop);
    segLoopTbl.Properties.VariableNames = varLoop;
    s(i).ReconSeg = segLoopTbl; 
end

sTestPAE = s;
clear s;

% Structure - Anomaly PAE
s = sTestPAE;
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

sTestAnomalyPAE = s;
clear s;

% Structure - Normal PAE
% Selecting the Test segments considered as an normal 
TestNormalPredSeg = segIdTest(indNormalPAETest);

s = sTestPAE;
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

sTestNormalPAE = s;
clear s;



TestPredFinalDouble = AnomalyTestTable.PAELogic;
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
textName= 'ConventionalAE';

 % sTrainPAE  structure - Reconstructed
structPAEName = strcat(textName,'-','TrainsCAE');
structPAEFileName = strcat(restPath,structPAEName);
save(structPAEFileName,'sTrainCAE');
% AnomalyTestTable
TestTableFileName= strcat(restPath, textName,'-TestAnomalyTable.csv');
writetable(AnomalyTestTable, TestTableFileName);
% MetricTable
metricTableFileName= strcat(restPath, textName,'-MetricTable.csv');
writetable( metricTable,metricTableFileName);
 % sTestPAE structure - Reconstructed
structPAEName = strcat(textName,'-','TestsCAE');
structPAEFileName = strcat(restPath,structPAEName);
save(structPAEFileName,'sTestCAE');
% sTestAnomalyPAE- Anomaly PAE structure
structAnomalyPAEName = strcat(textName,'-','TestsAnomalyPAE');
structAnomalyPAEFileName = strcat(restPath,structAnomalyPAEName);
save(structAnomalyPAEFileName,'sTestAnomalyCAE');
sTestNormalPAE- Noraml PAE structure
structNormalPAEName = strcat(textName,'-','TestsNormalCAE');
structNormalPAEFileName = strcat(restPath,structNormalPAEName);
save(structNormalPAEFileName,'sTestNormalCAE');
% Workspace variables
workspaceName= strcat(textName,'-','WorkSpaceVars');
workspaceFileName = strcat(restPath,workspaceName);
save(workspaceFileName)






