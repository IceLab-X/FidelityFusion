function xtime = heatedBlock2(lx,ly,theta, Hmax)
%%

% lx = 0.3
% ly = 0.2
% theta = pi/3
% Hmax = 0.8

% thermalmodelS = createpde('thermal','steadystate');
thermalmodelT = createpde('thermal','transient');
% thermalmodelT = 1

% Create a 2-D geometry by drawing one rectangle the size of the block and
% a second rectangle the size of the slot.
r1 = [3 4 -.5 .5 .5 -.5  -.8 -.8 .8 .8];
% r1 = [3 4 -1 1 1 -1  -1 -1 1 1];
% r2 = [3 4 -.05 .05 .05 -.05  -.4 -.4 .4 .4];
r2 = [4 0 0 lx ly  theta 0 0 0 0];
gdm = [r1; r2]';
%
% Subtract the second rectangle from the first to create the block with a
% slot.

g = decsg(gdm,'R1-R2',['R1'; 'R2']');

%
% Convert the decsg format into a geometry object. Include the geometry in
% the model.
geometryFromEdges(thermalmodelT,g);
%
% Plot the geometry with edge labels displayed. The edge labels will be
% used below in the function for defining boundary conditions.

% figure
% pdegplot(thermalmodelT,'EdgeLabels','on'); 
% axis([-.9 .9 -.9 .9]);
% title 'Block Geometry With Edge Labels Displayed'

% thermalBC(thermalmodelS,'Edge',1,'HeatFlux',-10);
% thermalBC(thermalmodelT,'Edge',3,'Temperature',100);

thermalProperties(thermalmodelT,'ThermalConductivity',1,...
                                'MassDensity',1,...
                                'SpecificHeat',1);

thermalBC(thermalmodelT,'Edge',1,'HeatFlux',-10);
thermalBC(thermalmodelT,'Edge',3,'Temperature',@transientBCHeatedBlock);




%% 
% Create a mesh with elements no larger than 0.2.
msh = generateMesh(thermalmodelT,'Hmax',Hmax);

% figure 
% pdeplot(thermalmodelT); 
% axis equal
% title 'Block With Finite Element Mesh Displayed'

%%
tlist = 0:.1:5;
thermalIC(thermalmodelT,0);
R = solve(thermalmodelT,tlist);
T = R.Temperature;

%% 
T = R.Temperature;
getClosestNode = @(p,x,y) min((p(1,:) - x).^2 + (p(2,:) - y).^2);

%%
% Call this function to get a node near the center of the right edge.
[~,nid] = getClosestNode( msh.Nodes, .5, 0 );

%%
% The two plots are shown side-by-side in the figure below. The temperature
% distribution at this time is very similar to that obtained from the
% steady-state solution above. At the right edge, for times less than about
% one-half second, the temperature is less than zero. This is because heat
% is leaving the block faster than it is arriving from the left edge. At
% times greater than about three seconds, the temperature has essentially
% reached steady-state.

% h = figure;
% h.Position = [1 1 2 1].*h.Position;
% subplot(1,2,1); 
% axis equal
% pdeplot(thermalmodelT,'XYData',T(:,end),'Contour','on','ColorMap','hot'); 
% axis equal
% title 'Temperature, Final Time, Transient Solution'
% subplot(1,2,2); 
% axis equal
% plot(tlist, T(nid,:)); 
% grid on
% title 'Temperature at Right Edge as a Function of Time';
% xlabel 'Time, seconds'
% ylabel 'Temperature, degrees-Celsius'


%% get final center solution
curve = T(nid,:);

delta_interpT = 0.0001;

yinterpol =interp1(tlist,curve,0:0.0001:5,'spline');
xtime = find(yinterpol>70);
xtime = xtime(1) * delta_interpT;

end