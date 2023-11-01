# stanford parser

cd stanford-parser-full/stanford-parser
java -mx1g edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat "typedDependencies" models/lexparser/englishPCFG.ser.gz snts.txt > snts_parse.txt