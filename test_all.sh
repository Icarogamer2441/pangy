pangy examples/hello.pgy -o hello
pangy examples/funcs.pgy -o funcs
pangy examples/vars.pgy -o vars
pangy examples/ifs.pgy -o ifs
pangy examples/imports/main.pgy -o imp_main
pangy examples/loops.pgy -o loops
pangy examples/classinclass.pgy -o classinclass
pangy examples/classinparams.pgy -o classinparams
pangy examples/exitfunc.pgy -o exitfunc
pangy examples/macros.pgy -o macros
pangy examples/macrosinclass.pgy -o macrosinclass
pangy examples/inputsandtypes.pgy -o inputsandtypes
pangy examples/files.pgy -o files
pangy examples/lists.pgy -o lists
pangy examples/matrixes.pgy -o matrixes
pangy examples/strcmp.pgy -o strcmp
pangy examples/args.pgy -o args
pangy examples/bitandshifts.pgy -o bitandshifts
pangy examples/string_idx.pgy -o string_idx
pangy examples/string_format.pgy -o string_format
pangy examples/listret.pgy -o listret

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

rm hello funcs vars ifs imp_main loops classinclass classinparams exitfunc macros macrosinclass inputsandtypes files lists matrixes strcmp args bitandshifts string_idx string_format listret
rm ./*.txt