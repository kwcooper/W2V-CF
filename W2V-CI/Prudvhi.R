########################################################
# Grammar for Prudvhi
########################################################

########################################################
# Clear the work environment
########################################################
rm(list = ls())

########################################################
# Get words into classes
########################################################
NOUN_HUMAN<- c("man","woman")
PRONOUN_HUMAN <- c("she","they")
NOUN_VEHICLE <- c("car","truck")
VERB_VEHICLE <- c("stop","break")
NOUN_DINNERWARE <- c("plate", "glass")
VERB_DINNERWARE <- c("smash","break")
NOUN_NEWS <- c("story", "news")
VERB_NEWS <- c("report","break")

########################################################
# Get a sentence
########################################################
Get_sentence <- function (s) {
  if (s == 1) Sentence <- c(sample(NOUN_HUMAN, 1), sample(VERB_VEHICLE, 1), sample(NOUN_VEHICLE, 1))
  if (s == 2) Sentence <- c(sample(NOUN_HUMAN, 1), sample(VERB_DINNERWARE, 1), sample(NOUN_DINNERWARE, 1))
  if (s == 3) Sentence <- c(sample(NOUN_HUMAN, 1), sample(VERB_NEWS, 1), sample(NOUN_NEWS, 1))
  return(Sentence)
}

########################################################
# Make a corpus
########################################################
N_sentences <- 100000
N_sentence_templates <- 2
Sentences <- matrix("", N_sentences)
for (i in 1:N_sentences) {
  Sentences[i] <- paste(Get_sentence(sample(N_sentence_templates,1)), collapse=" ")
}

########################################################
# Write the corpus to an external file
########################################################
fileConn<-file("Artificial_corpus.txt")
write(Sentences, fileConn) 
close(fileConn)
