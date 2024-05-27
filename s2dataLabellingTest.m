% Making Structures for Test semisupervised
% Data set 1:  Adenosine


clc; close all; clear;
filePathLabels = "Data\labels.csv";
SegmentLabels = readtable(filePathLabels,"PreserveVariable",true);

filePathBinaryLabels = "Data\labels2.csv";
BinaryLabels = readtable(filePathBinaryLabels,"PreserveVariable",true);

%Import Signals 
signalsFolder = dir("C:\Users\fesm704\OneDrive - The University of Auckland\DataBase\CNTFET35merAdenosine\P23\*-Test.csv");
signalsTables = table({signalsFolder.name}.',{signalsFolder.folder}.','VariableNames',{'name','folder'});

signalTitle = string(table2cell(signalsTables(:,1)));
signalDirectories = string(table2cell(signalsTables(:,2)));
signalDirectories = strcat(signalDirectories,'\',signalTitle);
signalTitleList = extractBefore(signalTitle, '-Test.csv');
signalNamePattern = "S" +  digitsPattern;
signalID= extract(signalTitle,signalNamePattern)';


vars = string(table2cell(SegmentLabels(:, "Channel"))');

ind = [];
for i=1: length(vars)
    indSigLoop = strcmp(vars(i), signalID);
    loopNonZero = find(1 == indSigLoop);
    ind = [ind; loopNonZero];
end


signalTitleNew = signalTitleList(ind);
signalsFolder = signalsFolder(ind,:);
signalsTables = signalsTables(ind,:);

% Making Structure
% The structure seoarate each signal to different segments and put all the similar segments to a table
analyteConcentration = ["0M","1nM","10nM","100nM","1uM","10uM"];


s = struct;

for k=1:length(analyteConcentration)
    s(k).AC = analyteConcentration(k);    
    s(k).DrainNA = table();
    s(k).zscore = table();
    s(k).SignalLabel = table();
    s(k).SegLabel = table();
    s(k).BiSegLabel = table();
end


vars = table2cell(SegmentLabels(:, "Channel"))';
for k=1:length(analyteConcentration)

    loopSegmentLabels = table2cell(SegmentLabels(:, k+1))';
    loopBinaryLabels = table2cell(BinaryLabels(:, k+1))';
    
    loopSegStr = string(table2cell(SegmentLabels(:, k+1))');
    loopBinaryStr = string(table2cell(BinaryLabels(:, k+1))');

    ind = loopSegStr ~= "";
    varLoop = vars(ind);
   
    loopSegmentTable = cell2table(loopSegmentLabels(ind), "VariableNames",varLoop);
    loopBinaryTable = cell2table(loopBinaryLabels(ind), "VariableNames",varLoop);
    s(k).SegLabel = loopSegmentTable;
    s(k).BiSegLabel = loopBinaryTable;    
end


for i = 1: size(signalsTables,1)

    signalTitle = string(table2cell(signalsTables(i,1)));
    signalDirectory = string(table2cell(signalsTables(i,2)));
    signalDirectory = strcat(signalDirectory,'\',signalTitle);
    signalTitle = extractBefore(signalTitle, '-Test.csv')

    signalNamePattern = "S" +  digitsPattern;
    signalID= extract(signalTitle,signalNamePattern);
    PoducerLabel = extractAfter(signalTitle, strcat(signalID,'-'));

    qualityLabelPattern = signalNamePattern+"-";
    signalSignalLabel = extract(signalTitle, qualityLabelPattern);
    signalSignalLabel = extractAfter(signalTitle,signalSignalLabel);

    T = readtable(signalDirectory,"PreserveVariableName",true);
    con = T.C_text;    
    concentrationSummary = groupsummary(T,"C_text");   


    % concentration analyte CA
    indexCA = [];
    for l=1:length(concentrationSummary.C_text)
        loopIndex = find(strcmp(concentrationSummary.C_text(l),analyteConcentration));
        indexCA = [indexCA loopIndex];
    end
    indexCA = sort(indexCA);
    concen = analyteConcentration(indexCA);



    for k = 1: length(indexCA)

        m= indexCA(k);
        CA = s(m).AC;
        idx = find(con == concen(k))
        idxSize = size(idx,1);

        LoopArrayRaw = T.DrainCurrentNanoA(idx);
        LoopArrayzscore = T.zscore(idx);

        LoopArraySignalLabel = PoducerLabel;


        LoopArrayRaw = LoopArrayRaw(end-499:end);
        LoopArrayzscore = LoopArrayzscore(end-499:end);


        LoopTableRaw = table(LoopArrayRaw,'VariableNames',signalID);
        LoopTablezscore = table(LoopArrayzscore,'VariableNames',signalID);
        LoopTableSignalLabel = table(LoopArraySignalLabel, 'VariableNames', signalID);

        CATableRaw = s(indexCA(k)).DrainNA;
        CATablezscore = s(indexCA(k)).zscore;
        CATableSignalLabel = s(indexCA(k)).SignalLabel;

        newTableRaw = [CATableRaw LoopTableRaw];
        newTablezscore = [CATablezscore  LoopTablezscore];
        newTableSignalLabel = [CATableSignalLabel LoopTableSignalLabel];

        s(m).DrainNA = newTableRaw;
        s(m).zscore = newTablezscore;
        s(m).SignalLabel = newTableSignalLabel;
    end
end


% Save Structure 
filepath = "DataTest";
structureName = strcat(filepath,'\','1AdeTestSemiSuperEricaVer3','.mat');

save(structureName,'s');

