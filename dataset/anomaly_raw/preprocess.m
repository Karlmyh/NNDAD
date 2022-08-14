FileData = load('ecoil.mat');
csvwrite('./csvfile/glass.csv', [FileData.X,FileData.y])

list=dir('*.mat')
for ii = 1:length(list)
    FileData=load(list(ii).name);
    datasetname=strsplit(list(ii).name,'.');
    
    FILENAME=strcat('./csvfile/',datasetname(1),'.csv')
    csvwrite(FILENAME{1,1}, [FileData.X,FileData.y])
    
end