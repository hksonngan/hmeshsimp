#include "random.h"
#include <iostream>

using namespace std;

int main()
{
	usrand(0);

	for (int i = 0; i < 200; i ++) {
		cout << urand() << endl;
	}
}