function iMode_freq = VibrationPlate(E,nu,m,hmax)

% This example shows how to calculate the vibration modes and frequencies
% of a 3-D simply supported, square, elastic plate. The dimensions and
% material properties of the plate are taken from a standard finite element
% benchmark problem published by NAFEMS, FV52 (<#15 [1]>).

iMode = 7;

N = 3;
model = createpde(N);
importGeometry(model,'Plate10x10x1.stl');

% Define the Coefficients in Toolbox Syntax
% Define the elastic modulus of steel, Poisson's ratio, and the material density.
% E = 200e9; % Modulus of elasticity in Pascals
% nu = .3; % Poisson's ratio
% m = 8000; % Material density in kg/m^3
%
% Incorporate these coefficients in toolbox syntax.
c = elasticityC3D(E,nu);
a = 0;

% Specify PDE Coefficients
% Include the PDE COefficients in |model|.
specifyCoefficients(model,'m',m,'d',0,'c',c,'a',a,'f',0);

applyBoundaryCondition(model,'mixed','Face',1:4,'u',0,'EquationIndex',3);

% hmax = 1; % Maximum element length for a moderately fine mesh
generateMesh(model,'Hmax',hmax,'GeometricOrder','quadratic');
% figure 
% pdeplot3D(model);
% title('Mesh with Quadratic Tetrahedral Elements');

%%
% maxLam = (1.1*refFreqHz(end)*2*pi)^2;
maxLam = (1.1*200*2*pi)^2;
r = [-.1 maxLam];
result = solvepdeeig(model,r);
eVec = result.Eigenvectors;
eVal = result.Eigenvalues;
numEig = size(eVal,1);
%
% Calculate the frequencies in Hz from the eigenvalues.
freqHz = sqrt(eVal(1:numEig))/(2*pi);

freqHz = real(freqHz)

% iMode_freq = freqHz(iMode);

try
    iMode_freq = freqHz(iMode);
catch
    fprintf('No valid iMode found...return the last one...')
    iMode_freq = freqHz(end);
    
end


% h = figure;
% h.Position = [100,100,900,600];
% numToPrint = min(length(freqHz),length(refFreqHz));
% for i = 4:numToPrint
%     subplot(4,2,i-3);
%     pdeplot3D(model,'ColorMapData',result.ModeShapes.uz(:,i));
%     axis equal
%     title(sprintf(['Mode=%d, z-displacement\n', ...
%     'Frequency(Hz): Ref=%g FEM=%g'], ...
%     i,refFreqHz(i),freqHz(i)));
% end

end