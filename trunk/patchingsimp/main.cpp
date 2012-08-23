extern int psimp_entry(int argc, char** argv, bool binary);

int main(int argc, char** argv)
{
	return psimp_entry(argc, argv, true);
}