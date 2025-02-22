import csv
import pandas as pd
from tabulate import tabulate
from scipy.stats import f, studentized_range, pearsonr
import numpy as np
import matplotlib.pyplot as plt

resultados = {}
datos = None

def cargar_datos():
    global datos
    nombre_archivo = "datos.csv"
    try:
        datos = pd.read_csv(nombre_archivo, encoding='utf-8')
        print("Datos cargados correctamente desde el archivo")
        return datos
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None
    
def guardar_datos():
    global datos
    nombre_archivo = "datos.csv"
    try:
        datos.to_csv(nombre_archivo, index=False, encoding='utf-8')
        print(f"Datos guardados correctamente en el archivo '{nombre_archivo}'.")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
    
def agregar_columna():
    global datos
    if datos is None:
        print("Primero debes cargar los datos.")
        return

    nombre_columna = input("Ingresa el nombre de la nueva columna: ")
    nueva_columna = []
    print("Ingresa 30 valores para la nueva columna:")
    for i in range(30):
        while True:
            try:
                valor = float(input(f"Valor {i + 1}: "))
                nueva_columna.append(valor)
                break
            except ValueError:
                print("Por favor, ingresa un numero valido.")

    if len(nueva_columna) == 30:
        datos[nombre_columna] = nueva_columna
        print(f"Columna '{nombre_columna}' añadida correctamente.")
        guardar_datos()
    else:
        print("Error: No se añadieron los valores correctamente.")


def calcular_estadisticas():
    global datos
    for columna in datos.columns:
        datos[columna + "^2"] = pow(datos[columna], 2)
    
    medias = round(datos.mean(numeric_only=True), 4)
    fila_medias = pd.DataFrame([medias], index=["Medias"])
    datos_media = pd.concat([datos, fila_medias])
    
    print("\nTabla de contingencia:")
    print(tabulate(datos_media, headers="keys", tablefmt="fancy_grid"))
    return datos

def analisis_andeva():
    global datos
    suma_totales = datos.sum(numeric_only=True)
    suma_cuadrados = (datos**2).sum(numeric_only=True)
    suma_x_pow2 = sum(pow(suma_totales[col], 2) for col in datos.columns)
    
    nt = datos.size
    t = len(datos.columns)
    
    C = pow(suma_totales.sum(), 2) / nt
    SCT = suma_cuadrados.sum() - C
    SCTR = suma_x_pow2 / 30 - C
    SCE = SCT - SCTR
    
    MCTR = SCTR / (t - 1)
    MCE = SCE / (nt - t)
    Fcalc = MCTR / MCE

    alpha = 0.05
    GL_tratamiento = t - 1
    GL_error = nt - t
    GL_total = GL_tratamiento + GL_error

    FTab = f.ppf(1 - alpha, dfn=GL_tratamiento, dfd=GL_error)
    FTab = round(FTab, 4)

    print("*" * 81)
    C = round(C, 4)
    SCT = round(SCT, 4)
    SCTR = round(SCTR, 4)
    SCE = round(SCE, 4)
    MCTR = round(MCTR, 4)
    MCE = round(MCE, 4)
    Fcalc = round(Fcalc, 4)

    print(f"C: {C}")
    print(f"SCT: {SCT}")
    print(f"SCTR: {SCTR}")
    print(f"SCE: {SCE}")
    print(f"MCTR: {MCTR}")
    print(f"MCE: {MCE}")
    print(f"Fcalc: {Fcalc}")
    print(f"FTab: {FTab}")

    tabla_datos = {
    "Fuente de Variación": ["Tratamiento", "Error", "Total"],
    "SC": [SCTR, SCE, SCTR + SCE],
    "GL": [GL_tratamiento, GL_error, GL_total],
    "MC": [MCTR, MCE, ""],
    "FCal (RV)": [Fcalc, "", ""]
    }

    tabla = pd.DataFrame(tabla_datos)
    print("*" * 81)
    print(tabla)
    print("*" * 81)
    
    if Fcalc > FTab:
        print("Conclusion: Rechazamos la hipotesis nula y se acepta la hipotesis alternativa")
    else:
        print("Conclusion: No hay evidencia suficiente para rechazar la hipotesis nula")

    x = np.linspace(0, FTab + 5, 1000)
    y = f.pdf(x, t, GL_error)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Distribución F", color="royalblue", lw=2)
    plt.fill_between(x, y, where=(x >= FTab), color='tomato', alpha=0.4, label="Region de rechazo", zorder=1)
    plt.axvline(FTab, color='red', linestyle='dashed', label=f"Ftab = {round(FTab, 4)}", lw=2, zorder=2)
    plt.axvline(Fcalc, color='lawngreen', linestyle='dashed', label=f"Fcal = {round(Fcalc, 4)}", lw=2, zorder=2)
    plt.fill_between(x, y, where=(x < FTab), color='palegreen', alpha=0.3, label="Region de aceptacion", zorder=0)
    plt.xlabel("Valores de F", fontsize=12)
    plt.ylabel("Densidad de probabilidad", fontsize=12)
    plt.title("Distribución F de Fisher", fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.show()

def generar_mapeo_nombres(datos):
    columnas = datos.columns

    mapeo_x_a_nombre = {f"x{i+1}": columna for i, columna in enumerate(columnas)}
    mapeo_nombre_a_x = {columna: f"x{i+1}" for i, columna in enumerate(columnas)}

    print("\nIdentificacion de variables:")
    for clave, valor in mapeo_x_a_nombre.items():
        print(f"{clave} -> {valor}")
    
    return mapeo_x_a_nombre, mapeo_nombre_a_x

def prueba_tukey(mapeo_nombre_a_x):
    global datos
    print("\nRealizando prueba de Tukey...")
    medias = datos.mean(numeric_only=True)
    t = len(medias)
    nt = datos.size
    GL_tratamiento = t - 1
    GL_error = nt - t
    suma_cuadrados_totales = (datos**2).sum().sum()
    suma_cuadrados_columnas = sum(datos.sum()**2 / len(datos))
    SCE = suma_cuadrados_totales - suma_cuadrados_columnas
    MCE = SCE / GL_error

    ni = len(datos)
    alpha = 0.05
    q = studentized_range.ppf(1 - alpha, GL_tratamiento + 1, GL_error)
    DHS = q * (MCE / ni) ** 0.5
    DHS = round(DHS, 4)

    variables = list(medias.index)
    tabla_tukey = pd.DataFrame(np.empty((t, t), dtype=object), index=variables, columns=variables)

    for i in range(t):
        for j in range(i + 1, t):
            dif = round(medias.iloc[i] - medias.iloc[j], 4)
            tabla_tukey.iloc[i, j] = dif

    print(f"\nDHS calculado: {DHS}")
    print("\nTabla de diferencias de Tukey:")
    print(tabulate(tabla_tukey, headers="keys", tablefmt="fancy_grid"))

    independientes = []
    for fila in variables:
        for columna in variables:
            if fila != columna:
                if isinstance(tabla_tukey.loc[fila, columna], float):
                    if tabla_tukey.loc[fila, columna] > DHS:
                        clave_fila = mapeo_nombre_a_x[fila]
                        clave_columna = mapeo_nombre_a_x[columna]
                        independientes.append((clave_fila, clave_columna))

    print("\nPares de variables linealmente independientes:")
    for par in independientes:
        print(f"{par[0]} y {par[1]} son linealmente independientes!!")

    resultados['tabla_tukey'] = tabla_tukey
    resultados['DHS'] = DHS
    resultados['independientes'] = independientes

    return tabla_tukey, DHS, independientes

def calcular_correlaciones(mapeo_x_a_nombre):
    global datos
    if 'independientes' not in resultados:
        print("Error: No se han calculado las variables independientes. Ejecuta la Prueba de Tukey primero")
        return

    independientes = resultados['independientes']
    print("\nCalculando correlaciones lineales...")

    correlaciones = {}
    for par in independientes:
        var1 = mapeo_x_a_nombre[par[0]]
        var2 = mapeo_x_a_nombre[par[1]]

        r, p_value = pearsonr(datos[var1], datos[var2])
        correlaciones[f"{var1} y {var2}"] = (round(r, 4), round(p_value, 4))

    print("\nCorrelaciones lineales calculadas:")
    for par, (r, p) in correlaciones.items():
        print(f"Correlacion entre {par}: r = {r}")

    resultados['correlaciones'] = correlaciones
    return correlaciones

def gauss_jordan(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)
    matriz = np.hstack((A, B))  #matriz aumentada [A|B]
    
    n = len(matriz)
    
    for i in range(n):
        #verifica si el pivote es cero e intercambia filas si es necesario
        if matriz[i][i] == 0:
            for k in range(i + 1, n):
                if matriz[k][i] != 0:
                    matriz[[i, k]] = matriz[[k, i]]  #intercambia filas
                    break
        
        #hace que el pivote sea 1 dividiendo toda la fila por el pivote
        pivote = matriz[i][i]
        matriz[i] = matriz[i] / pivote
        
        #hace ceros en el resto de la columna
        for j in range(n):
            if j != i:
                factor = matriz[j][i]
                matriz[j] -= factor * matriz[i]
    
    #retorna solo la ult columnaf
    soluciones = matriz[:, -1]
    return soluciones, matriz

def regresion_multiple():
    global datos
    if datos is None:
        print("Primero debes cargar los datos")
        return
    
    independientes = resultados['independientes']

    if 'independientes' not in resultados:
        print("Error: No se han calculado las variables independientes. Ejecuta la Prueba de Tukey primero.")
        return

    independientes = resultados['independientes']
    if len(independientes) > 2:
        print("xd")
        return

    try:
        datos_tabla2 = datos.rename(columns={
            "Temperatura": "x1",
            "Presion": "x2",
            "Humedad": "y"
        })
    except KeyError as e:
        print(f"Error: No se encontró la columna esperada en los datos. Detalles: {e}")
        return

    datos_tabla2["x1i . x2i"] = round(datos_tabla2["x1"] * datos_tabla2["x2"], 4)
    datos_tabla2["x1i . yi"] = round(datos_tabla2["x1"] * datos_tabla2["y"], 4)
    datos_tabla2["x1i^2"] = round(pow(datos_tabla2["x1"], 2), 4)
    datos_tabla2["x2i . yi"] = round(datos_tabla2["x2"] * datos_tabla2["y"], 4)
    datos_tabla2["x2i^2"] = round(pow(datos_tabla2["x2"], 2), 4)

    columnas_tabla2 = [
        "y", "x1", "x2", "x1i^2",
        "x1i . x2i", "x1i . yi",
        "x2i^2", "x2i . yi"
    ]
    estructura_tabla = datos_tabla2[columnas_tabla2]

    sumatorias = round(estructura_tabla.sum(numeric_only=True), 4)
    fila_sumatorias = pd.DataFrame([sumatorias], index=["Total"])

    tabla_final2 = pd.concat([estructura_tabla, fila_sumatorias])
    tabla2 = tabulate(
        tabla_final2,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=True
    )

    print("\nTabla para el sistema de ecuaciones:")
    print(tabla2)

    n = len(datos)
    sum_y = sumatorias["y"]
    sum_x1 = sumatorias["x1"]
    sum_x2 = sumatorias["x2"]
    sum_x1_y = sumatorias["x1i . yi"]
    sum_x2_y = sumatorias["x2i . yi"]
    sum_x1_x2 = sumatorias["x1i . x2i"]
    sum_x1pow2 = sumatorias["x1i^2"]
    sum_x2pow2 = sumatorias["x2i^2"]

    A = [
        [n, sum_x1, sum_x2],
        [sum_x1, sum_x1pow2, sum_x1_x2],
        [sum_x2, sum_x1_x2, sum_x2pow2]
    ]

    B = [
        sum_y,
        sum_x1_y,
        sum_x2_y
    ]

    soluciones, matriz_final = gauss_jordan(A, B)

    coeficientes = ["β0", "β1", "β2"]
    tabla_resultados = [[coef, valor] for coef, valor in zip(coeficientes, soluciones)]

    print("\nResultados de la regresion multiple:")
    print(tabulate(tabla_resultados, headers=["Coeficiente", "Valor"], tablefmt="fancy_grid"))

    print("\nMatriz aumentada:")
    print(tabulate(matriz_final, tablefmt="fancy_grid"))

    calcular_y_regresion(soluciones)

def calcular_y_regresion(soluciones):
    print("\nModelo de regresion multiple: Y = β0 + β1X1 + β2X2")
    Y = lambda x1, x2: soluciones[0] + soluciones[1] * x1 + soluciones[2] * x2
    ans = input("¿Desea ingresar valores para X1 y X2? (1) Sí / (Otro) No: ")

    while ans == "1":
        try:
            x1 = float(input("Introduce el valor de X1: "))
            x2 = float(input("Introduce el valor de X2: "))
            resultado = Y(x1, x2)
            print(f"Resultado: Y = {round(resultado, 4)}")
        except ValueError:
            print("Por favor, introduce valores numericos validos.")
        
        ans = input("¿Desea ingresar mas valores? (1) Si / (Otro) No: ")

def calcular_y_regresionL(a, b):
    print("\nY en regresion lineal: Y = a + bX")
    Y = lambda x: a + b * x
    ans = input("¿Desea ingresar algun valor de X en Y? (1) Sí / (Otro) No: ")

    while ans == "1":
        try:
            numX = float(input("Introduzca el Valor de X: "))
            print(f"El Resultado es: Y = {round(Y(numX), 4)}")
        except ValueError:
            print("Introduzca un valor numerico valido!")

        ans = input("¿Desea ingresar algun valor de X en Y? (1) Sí / (Otro) No: ")

def regresion_lineal_independiente(mapeo_x_a_nombre):
    global datos
    if 'independientes' not in resultados:
        print("Error: No se han calculado las variables independientes. Ejecuta la Prueba de Tukey primero")
        return

    independientes = resultados['independientes']
    print("\nGenerando graficas de regresion lineal...")

    def regresion_lineal(x, y):
        n = len(x)
        sumX = x.sum()
        sumY = y.sum()
        sumX2 = (x**2).sum()
        sumXY = (x * y).sum()

        b = ((sumXY) - (sumX * sumY) / n) / (sumX2 - (sumX**2) / n)
        a = (sumY - (sumX * b)) / n
        return round(a, 4), round(b, 4)

    def graficar_regresion(x, y, a, b, titulo):
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Datos')
        plt.plot(x, a + b * x, color='red', label=f'Regresion: y = {a} + {b}x')
        plt.title(titulo)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    for var1, var2 in independientes:
        if var1 in mapeo_x_a_nombre and var2 in mapeo_x_a_nombre:
            var1_real = mapeo_x_a_nombre[var1]
            var2_real = mapeo_x_a_nombre[var2]

            x = datos[var1_real]
            y = datos[var2_real]
            a, b = regresion_lineal(x, y)
            print(f"\nRegresion lineal para {var1_real} y {var2_real}: y = {a} + {b}x")
            graficar_regresion(x, y, a, b, f"Regresion lineal: {var1_real} vs {var2_real}")
            calcular_y_regresionL(a, b)
        else:
            print(f"Error: No se encontro el mapeo para las variables {var1} o {var2}")

def menu_principal():
    global datos
    mapeo_x_a_nombre = None
    mapeo_nombre_a_x = None

    datos = cargar_datos()

    if datos is not None:
        mapeo_x_a_nombre, mapeo_nombre_a_x = generar_mapeo_nombres(datos)

    while True:
        print("\nMenu principal:")
        print("1. Mostrar datos")
        print("2. Tabla de contingencia")
        print("3. Realizar analisis ANDEVA")
        print("4. Prueba de Tukey")
        print("5. Realizar correlacion y regresion lineal")
        print("6. Regresión multiple")
        print("7. Agregar columna")
        print("8. Salir")
        
        opcion = input("Selecciona una opcion: ")
        
        if opcion == "1":
            cargar_datos()
            if datos is not None:
                print(tabulate(datos, headers="keys", tablefmt="fancy_grid", showindex=True))
        elif opcion == "2":
            cargar_datos()
            if datos is not None:
                calcular_estadisticas()
            else:
                print("Primero debes cargar los datos")
        elif opcion == "3":
            cargar_datos()
            if datos is not None:
                datos = analisis_andeva()
            else:
                print("Primero debes cargar los datos")
        elif opcion == "4":
            cargar_datos()
            if datos is not None:
                prueba_tukey(mapeo_nombre_a_x)
            else:
                print("Primero debes cargar los datos")
        elif opcion == "5":
            cargar_datos()
            if datos is not None:
                calcular_correlaciones(mapeo_x_a_nombre)
                regresion_lineal_independiente(mapeo_x_a_nombre)
            else:
                print("Primero debes cargar los datos")
        elif opcion == "6":
            cargar_datos()
            if datos is not None:
                regresion_multiple()
            else:
                print("Primero debes cargar los datos")
        elif opcion == "7":
            cargar_datos()
            if datos is not None and mapeo_x_a_nombre is not None:
                agregar_columna()
                if datos is not None:
                    mapeo_x_a_nombre, mapeo_nombre_a_x = generar_mapeo_nombres(datos)
            else:
                print("Primero debes cargar los datos")
        elif opcion == "8":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no valida. Por favor, selecciona una opcion del menu")

menu_principal()
