% VibrationPlate


E = 200e9; % Modulus of elasticity in Pascals. 100e9<E<500e9
nu = .3; % Poisson's ratio                      0.2<nu<0.6
m = 8000; % Material density in kg/m^3          6000<m<10000

% Low-fidellity
hmax = 1.2;
lf = VibrationPlate(E,nu,m,hmax);


% High-fidellity
hmax = 0.6;
hf = VibrationPlate(E,nu,m,hmax);