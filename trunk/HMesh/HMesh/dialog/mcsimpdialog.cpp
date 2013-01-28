#include "mcsimpdialog.h"
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QHBoxLayout>

QMCSimpDialog::QMCSimpDialog() {
	//fileLabel = new QLabel(tr("volume file ..."));
	//isoValueLabel = new QLabel(tr("Iso Value"));
	//decimateRateLabel = new QLabel(tr("Decimate Rate"));
	//openFileButton = new QPushButton(tr("Open"));
	//decimateButton = new QPushButton(tr("Decimate"));

	//QVBoxLayout *leftLayout = new QVBoxLayout();
	//leftLayout->addWidget(openFileButton);
	//leftLayout->addWidget(isoValueLabel);
	//leftLayout->addWidget(decimateRateLabel);

	//QVBoxLayout *rightLayout = new QVBoxLayout();
	//rightLayout->addWidget(fileLabel);
	////leftLayout->addWidget();

	//QHBoxLayout *topLayout = new QHBoxLayout();
	//topLayout->addLayout(leftLayout);
	//topLayout->addLayout(rightLayout);

	//QHBoxLayout *bottomLayout = new QHBoxLayout();
	//bottomLayout->addStretch();
	//bottomLayout->addWidget(decimateButton);

	//QVBoxLayout *mainLayout = new QVBoxLayout();
	//mainLayout->addLayout(topLayout);
	//mainLayout->addLayout(bottomLayout);

	//this->setLayout(mainLayout);
	//
	//setWindowTitle(tr("Params"));
	//setFixedHeight(sizeHint().height()); 

	setupUi(this);
}

QMCSimpDialog::~QMCSimpDialog() {
	
}