#include <iostream>

using namespace std;

class A {
public:
	virtual void func1() {
		cout << "A::func1" << endl;
	}
	void func2() {
		cout << "A::func2" << endl;
		func1();
	}
	void func3() {
		cout << "A::func3" << endl;
		func1();
	}
	void func4() {
		cout << "A::func4" << endl;
	}
};

class B: public A {
public:
	virtual void func1() {
		cout << "B::func1" << endl;
		A::func1();
	}
	void func3() {
		cout << "B::func3" << endl;
		A::func3();
		func2();
		func4();
	}
	void func4() {
		cout << "B::func4" << endl;
	}
};

int main() {
	B b;

	b.func1();
	//b.func3();
	//cout << endl;
	//b.func2();
}