#include "qslimdialog.h"
#include <QtGui/QVBoxLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QSlimDialog::QSlimDialog(QWidget *parent)
	:QDialog(parent)
{
	label = new QLabel(tr("parameters:"));
	lineEdit = new QLineEdit;
	label->setBuddy(lineEdit);

	//caseCheckBox = new QCheckBox(tr("Match &case"));
	//backwardCheckBox = new QCheckBox(tr("Search &backford"));

	goButton = new QPushButton(tr("go"));
	goButton->setDefault(true);
	goButton->setEnabled(true);
	goButton->setFixedWidth(60);

	//closeButton = new QPushButton(tr("Close"));

	//connect(lineEdit, SIGNAL(textChanged(const QString&)), this, SLOT(enableFindButton(const QString&)));
	connect(goButton, SIGNAL(clicked()), this, SLOT(goClicked()));
	//connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));

	QVBoxLayout *leftLayout = new QVBoxLayout;
	leftLayout->addWidget(label);
	leftLayout->addStretch();
	QVBoxLayout *rightLayout = new QVBoxLayout;
	rightLayout->addWidget(lineEdit);
	rightLayout->addWidget(goButton);

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addLayout(leftLayout);
	mainLayout->addLayout(rightLayout);
	
	this->setLayout(mainLayout);

	setWindowTitle(tr("QSlim"));
	setFixedHeight(sizeHint().height()); 
}

QSlimDialog::~QSlimDialog()
{

}

void QSlimDialog::goClicked()
{
	this->accept();
}