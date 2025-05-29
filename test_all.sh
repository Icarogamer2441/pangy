pangy compile examples/hello.pgy -o hello
pangy compile examples/funcs.pgy -o funcs
pangy compile examples/vars.pgy -o vars
pangy compile examples/ifs.pgy -o ifs
pangy compile examples/imports/main.pgy -o imp_main
pangy compile examples/loops.pgy -o loops
pangy compile examples/classinclass.pgy -o classinclass
pangy compile examples/classinparams.pgy -o classinparams
pangy compile examples/exitfunc.pgy -o exitfunc
pangy compile examples/macros.pgy -o macros
pangy compile examples/macrosinclass.pgy -o macrosinclass
pangy compile examples/inputsandtypes.pgy -o inputsandtypes
pangy compile examples/files.pgy -o files
pangy compile examples/lists.pgy -o lists
pangy compile examples/matrixes.pgy -o matrixes
pangy compile examples/strcmp.pgy -o strcmp
pangy compile examples/args.pgy -o args
pangy compile examples/bitandshifts.pgy -o bitandshifts
pangy compile examples/string_idx.pgy -o string_idx
pangy compile examples/string_format.pgy -o string_format
pangy compile examples/listret.pgy -o listret
pangy compile examples/publicprivate.pgy -o publicprivate
pangy compile examples/execterm.pgy -o execterm
pangy compile test/string_utils_test.pgy -o string_utils
pangy compile test/math_utils_test.pgy -o math_utils
pangy compile examples/changeidxval.pgy -o changeidxval
pangy compile examples/insert.pgy -o insert
pangy compile examples/constructor.pgy -o constructor
pangy compile examples/floats.pgy -o floats
pangy compile examples/float_lists_advanced.pgy -o float_lists_advanced
pangy compile examples/float_lists.pgy -o float_lists
pangy compile examples/float_conversions.pgy -o float_conversions

echo "Running hello..."
./hello
echo "--------------------------------"
echo "Running funcs..."
./funcs
echo "--------------------------------"
echo "Running vars..."
./vars
echo "--------------------------------"
echo "Running ifs..."
./ifs
echo "--------------------------------"
echo "Running imp_main..."
./imp_main
echo "--------------------------------"
echo "Running loops..."
./loops
echo "--------------------------------"
echo "Running classinclass..."
./classinclass
echo "--------------------------------"
echo "Running classinparams..."
./classinparams
echo "--------------------------------"
echo "Running exitfunc..."
./exitfunc
echo "--------------------------------"
echo "Running macros..."
./macros
echo "--------------------------------"
echo "Running macrosinclass..."
./macrosinclass
echo "--------------------------------"
echo "Running inputsandtypes..."
./inputsandtypes
echo "--------------------------------"
echo "Running files..."
./files
echo "--------------------------------"
echo "Running lists..."
./lists
echo "--------------------------------"
echo "Running matrixes..."
./matrixes
echo "--------------------------------"
echo "Running strcmp..."
./strcmp
echo "--------------------------------"
echo "Running args..."
./args testing "this is a test" abc
echo "--------------------------------"
echo "Running bitandshifts..."
./bitandshifts
echo "--------------------------------"
echo "Running string_idx..."
./string_idx
echo "--------------------------------"
echo "Running string_format..."
./string_format
echo "--------------------------------"
echo "Running listret..."
./listret
echo "--------------------------------"
echo "Running publicprivate..."
./publicprivate
echo "--------------------------------"
echo "Running execterm..."
./execterm
echo "--------------------------------"
echo "--------------------------------"
echo "Running changeidxval..."
./changeidxval
echo "--------------------------------"
echo "Running insert..."
./insert
echo "--------------------------------"
echo "Running constructor..."
./constructor
echo "--------------------------------"
echo "Running floats..."
./floats
echo "--------------------------------"
echo "Running float_lists_advanced..."
./float_lists_advanced
echo "--------------------------------"
echo "Running float_lists..."
./float_lists
echo "--------------------------------"
echo "Running float_conversions..."
./float_conversions
echo "--------------------------------"
echo "========= TESTING UTILS ========="
echo "--------------------------------"
echo "Running string_utils..."
./string_utils
echo "--------------------------------"
echo "Running math_utils..."
./math_utils

rm hello funcs vars ifs imp_main loops classinclass classinparams exitfunc macros macrosinclass inputsandtypes files lists matrixes strcmp args bitandshifts string_idx string_format listret publicprivate execterm string_utils math_utils changeidxval insert constructor floats float_lists_advanced float_lists float_convertions
rm ./*.txt