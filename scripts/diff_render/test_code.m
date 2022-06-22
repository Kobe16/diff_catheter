clc
clear all
close all

cylinder = load('cylinder_primitive.csv');

plot3(cylinder(:,1), cylinder(:,2), cylinder(:,3), '.-')
daspect([1 1 1])
grid on

hold on
plot3(cylinder(1,1), cylinder(1,2), cylinder(1,3), 'r*')
plot3(cylinder(2,1), cylinder(2,2), cylinder(2,3), 'g<')
plot3(cylinder(3,1), cylinder(3,2), cylinder(3,3), 'b<')




