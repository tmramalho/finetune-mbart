from fairseq.models.bart import BARTModel
checkpoint_folder=''
bart = BARTModel.from_pretrained(
        f'{checkpoint_folder}/mbart.cc25',
        checkpoint_file='jaen88.pt',
        bpe='sentencepiece',
        sentencepiece_vocab=f'{checkpoint_folder}/mbart.cc25/sentence.bpe.model')
bart.eval()  # disable dropout (or leave in train mode to finetune)

# tokens = bart.encode('ご飯を食べない女房の正体は、頭に口がある化物。')
sentence_list = ['旅行に来る外国人はこれからも少ないままになりそうです。このため、日本の経済はとても厳しくなっています。']
translation = bart.sample(sentence_list, beam=5)
print(translation)
breakpoint()

#sample does not properly split multiple sentences with </s> for bpe
# also the [en_XX] token is not added at the end of a sentence, but doesn't seem to matter much?
