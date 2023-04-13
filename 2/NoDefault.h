class NoDefault {
public:
    NoDefault(int) {};
};
class C {
public:
    C() : my_mem(0) {}
private:
    NoDefault my_mem;
};