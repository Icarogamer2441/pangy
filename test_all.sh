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

rm hello funcs vars ifs imp_main loops classinclass classinparams exitfunc macros macrosinclass