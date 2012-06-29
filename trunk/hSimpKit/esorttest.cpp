#include "h_esort.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#define MASK 0xff

class IntRecord : public hERadixRecord {
public:
	virtual unsigned int getDigit(int which_digit) const;
	virtual int getDigitCount() const;
	virtual int getDigitSize(int which_digit) const;

	virtual bool read(FILE *fp);
	virtual bool write(FILE *fp);
	virtual bool operator <(const IntRecord &r) const;
	virtual bool operator >(const IntRecord &r) const;

public:
	unsigned int val;
	float f[10];
};

unsigned int IntRecord::getDigit(int which_digit) const {
	
	unsigned int i;

	i = val & (MASK << (which_digit * 8));
	i >>= which_digit * 8;

	return i;
}

int IntRecord::getDigitCount() const {
	return 4;
}

int IntRecord::getDigitSize(int which_digit) const {
	return 256;
}

bool IntRecord::read(FILE *fp) {
	if (fscanf(fp, "%d", &val) == EOF) {
		return false;
	}

	return true;
}

bool IntRecord::write(FILE *fp) {
	fprintf(fp, "%d\n", val);

	return true;
}

bool IntRecord::operator <(const IntRecord &r) const {
	return val < r.val;
}

bool IntRecord::operator >(const IntRecord &r) const {
	return val > r.val;
}

#define N 105
#define R 10

int main() {
	int i;

	FILE* f = fopen("nums.txt", "w");

	srand ( time(NULL) );

	for (i = 0; i < 105; i ++) {
		fprintf(f, "%d\n", (unsigned)rand() * (unsigned int)rand());
	}

	fclose(f);

	f = fopen("nums.txt", "r");

	hERadixSort<IntRecord>(f, "nums.txt", "sorted_nums.txt", R, N);
}