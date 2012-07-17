#include <iostream>
#include <fstream>

using std::ofstream;

int main(int argc, char** argv)
{
	ofstream fout("test.txt");
	ofstream* fp = &fout;

	*fp << "hhhhhhhh" << std::endl;
	fp->write("OOOOOO", strlen("OOOOOO"));

	return 0;
}

