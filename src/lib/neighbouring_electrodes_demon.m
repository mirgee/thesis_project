%% Workout neighbouring electrodes (inter-electrode distances)
% Nik Murphy, 2016.  
% Lewybody Lab, Newcastle University, Institute of Neuroscience
% (http://www.lewybodylab.org/)

%DISCLAIMER: The solution below is a demonstration and might require
%ammendments to suit your data.

%Arc length method assumes a centroid at (xo,yo,zo) = (0,0,0)
%and input values are in rectangular Cartesian coordinates.
% X Y Z values
%LOAD AND CREATE SPLINE FILES USING CSD TOOLBOX
%(http://psychophysiology.cpmc.columbia.edu/software/CSDtoolbox/index.html, Kayser et al, 2006)
load('E:\functions\CSDtoolbox\128_electrodes.mat');
M = ExtractMontage('10-5-System_Mastoids_EGI129.csd',trodes);
MapMontage(M);
% Create spline co-ordinates
[G,H] = GetGH(M);

%setting up electrodes - EEGLAB read locations
%(https://sccn.ucsd.edu/eeglab/ Delorme & Makeig, 2004)
elNum=textread(strcat('ElectrodesLabelsNum128.prn'),'%3s');% list of EEG electrodes numbers
electr = readlocs('standard_128_1005.elc'); % reading electrode numbs from this source
electr(1)=[];  %deleting first electrode - ground
FN = fieldnames(electr);
for loopIndex = 1:size(electr,2)
    a = struct2cell(electr(loopIndex));
    X(loopIndex,1) = cell2mat(a(2,1));
    Y(loopIndex,1) = cell2mat(a(3,1));
    Z(loopIndex,1) = cell2mat(a(4,1));
end

%Workout generalised distance between electrodes based on average head
%circumference across all subjects ( Author: David Groppe, Kutaslab,
%5/2011, http://openwetware.org/wiki/Mass_Univariate_ERP_Toolbox)
chan_hood=spatial_neighbors(electr,60,101.641097); %max dist (cartesian [5cm]), radius (cartesian)
%this estimates which electrodes are within 5cm of eachother.

% Create neighbourhood information
%%% This information will be used to work out what electrodes are within
%%% distance of each electrode for cluster formation.
clusterIndices = cell(128,1);
clusterLabels = cell(128,1);
for iter = 1:128
    f = find(chan_hood(iter,:)==1);
    clusterIndices{iter,1}(1:size(f,2)) = f;
    clusterLabels{iter,1}(1:size(f,2)) = elecs(1,f);
    clear f
end