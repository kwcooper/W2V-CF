########################################################
# Make a corpus from the Elman (1990) grammar
########################################################

########################################################
# Clear the work environment
########################################################
rm(list = ls())

########################################################
# Get words into classes
########################################################
NOUN_HUM <- c("man","woman")
NOUN_ANIM <- c("cat","mouse")
NOUN_INANIM <- c("book","rock")
NOUN_AGRESS <- c("dragon","monster")
NOUN_FRAG <- c("glass","plate")
NOUN_FOOD <- c("cookie","cake")     # break --> cake
VERB_INTRAN <- c("think","sleep")
VERB_TRAN <- c("see","chase")
VERB_AGPAT <- c("move","break")
VERB_PERCEPT <- c("smell","see")
VERB_DESTROY <- c("break","smash")
VERB_EAT <- c("eat")

########################################################
# Get a sentence
########################################################
Get_sentence <- function (s) {
  if (s == 1)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_EAT, 1), sample(NOUN_FOOD, 1))
  if (s == 2)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_PERCEPT, 1), sample(NOUN_INANIM, 1))
  if (s == 3)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_DESTROY, 1), sample(NOUN_FRAG, 1))
  if (s == 4)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_INTRAN, 1))
  if (s == 5)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_TRAN, 1), sample(NOUN_HUM, 1))
  if (s == 6)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_AGPAT, 1), sample(NOUN_INANIM, 1))
  if (s == 7)  Sentence <- c(sample(NOUN_HUM, 1), sample(VERB_AGPAT, 1))
  if (s == 8)  Sentence <- c(sample(NOUN_ANIM, 1), sample(VERB_EAT, 1), sample(NOUN_FOOD, 1))
  if (s == 9)  Sentence <- c(sample(NOUN_ANIM, 1), sample(VERB_TRAN, 1), sample(NOUN_ANIM, 1))
  if (s == 10) Sentence <- c(sample(NOUN_ANIM, 1), sample(VERB_AGPAT, 1), sample(NOUN_INANIM, 1))
  if (s == 11) Sentence <- c(sample(NOUN_ANIM, 1), sample(VERB_AGPAT, 1))
  if (s == 12) Sentence <- c(sample(NOUN_INANIM, 1), sample(VERB_AGPAT, 1))
  if (s == 13) Sentence <- c(sample(NOUN_AGRESS, 1), sample(VERB_DESTROY, 1), sample(NOUN_FRAG, 1))
  if (s == 14) Sentence <- c(sample(NOUN_AGRESS, 1), sample(VERB_EAT, 1), sample(NOUN_HUM, 1))
  if (s == 15) Sentence <- c(sample(NOUN_AGRESS, 1), sample(VERB_EAT, 1), sample(NOUN_ANIM, 1))
  if (s == 16) Sentence <- c(sample(NOUN_AGRESS, 1), sample(VERB_EAT, 1), sample(NOUN_FOOD, 1))
  return(Sentence)
}

########################################################
# Make a corpus
########################################################
N_sentences <- 10000
Sentences <- matrix("", N_sentences)
for (i in 1:N_sentences) {
  Sentences[i] <- paste(Get_sentence(sample(16, 1)), collapse=" ")
}

########################################################
# Write the corpus to an external file
########################################################
fileConn<-file("Elman_corpus.txt")
  write(Sentences, fileConn) 
close(fileConn)


