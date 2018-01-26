tfs <- read.csv('~/Desktop/tmp/technical_factors.csv', header=FALSE)
ths <- read.csv('~/Desktop/tmp/technical_headers.txt', header=FALSE)
ths <- unlist(ths, use.names = FALSE)
colnames(tfs) <- ths

X <- read.csv('~/Desktop/tmp/X-Lung.csv', header=FALSE)
Y <- read.csv('~/Desktop/tmp/Y-Lung-retrained-mean-256.csv', header=FALSE)

donorIDs <- read.csv('~/Desktop/tmp/donorIDs.txt', header=FALSE)
donorIDs <- unlist(donorIDs, use.names = FALSE)

has_tf <- read.csv('~/Desktop/tmp/has_tf.txt', header=FALSE)
has_tf <- as.logical(unlist(has_tf, use.names = FALSE))


