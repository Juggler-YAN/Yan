#ifndef GRAPH_H
#define GRAPH_H

#include <string>

using namespace std;

static const double PI = 3.1415926;

class Shape {
public:
    virtual string name() const = 0;
    virtual ~Shape() {}
};

class Shape_2D : public Shape {
public:
    virtual double perimeter() const = 0;
    virtual double area() const = 0;
    ~Shape_2D() override {}
};

class Shape_3D : public Shape {
public:
    virtual double volume() const = 0;
    ~Shape_3D() override {}
};

class Box : public Shape_2D {
public:
    Box() = default;
    Box(double x, double y) : len_x(x), len_y(y) {}
    string name() const override { return string("Box"); }
    double perimeter() const override { return (len_x + len_y) * 2; }
    double area() const override { return len_x * len_y; }
    ~Box() override {}
private:
    double len_x;
    double len_y;
};

class Circle : public Shape_2D {
public:
    Circle() = default;
    Circle(double r) : radius(r) {}
    string name() const override { return string("Circle"); }
    double perimeter() const override { return 2 * PI * radius; }
    double area() const override { return PI * radius * radius; }
    ~Circle() override {}
private:
    double radius;
};

class Sphere : public Shape_3D {
public:
    Sphere() = default;
    Sphere(double r) : radius(r) {}
    string name() const override { return string("Sphere"); }
    double volume() const override { return 4.0 / 3 * PI * radius * radius * radius; }
    ~Sphere() override {}
private:
    double radius;
};

class Cone : public Shape_3D {
public:
    Cone() = default;
    Cone(double r, double h) : bottomradius(r), height(h) {}
    string name() const override { return string("Cone"); }
    double volume() const override { return 1.0 / 3 * PI * bottomradius * bottomradius * height; }
    ~Cone() override {}
private:
    double bottomradius;
    double height;
};

#endif