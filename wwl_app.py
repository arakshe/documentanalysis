from worldwarletters import WorldWarLetters
import pprint as pp
import wwl_parsers as tp

def main():
    tt = WorldWarLetters()
    tt.load_text('letter1.txt', 'A')
    tt.load_text('letter2.txt', 'B')
    tt.load_text('letter3.txt', 'C')
    tt.load_text('letter4.txt', 'D')
    tt.load_text('letter5.txt', 'E')
    tt.load_text('myfile.json',  'J', parser=tp.json_parser)
    pp.pprint(tt.data)
    tt.compare_num_words()

if __name__ == '__main__':
    main()
