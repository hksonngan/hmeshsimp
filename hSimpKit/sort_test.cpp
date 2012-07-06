#include "random.h"
#include <iostream>
#include "vec3.h"

using namespace std;

int main()
{
	usrand(0);

	Vec3 v;

	for (int i = 0; i < 200; i ++) {
		cout << urand() << endl;
	}
}