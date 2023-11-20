"""import ./mes_fonctions
print(pwd())
println(f("X"))

println(predire_val_arbre("PEUGEOT","206+","COMPACT",1.1,4,"jsp","X",9,6,8))


"""
#http://localhost:8050/

using MLJ
using JLD2
using DataFrames, PlotlyJS, CSV, RDatasets

df = CSV.read("D:/Master/S7/R-Julia/Projet/CO2 Emissions_Canada.csv", DataFrames.DataFrame)
rename!(df, :"CO2 Emissions(g/km)"=> "CO2")
rename!(df, :"Fuel Consumption Comb (L/100 km)"=>"Comb")
rename!(df, :"Fuel Type"=>"Fuel")
p1 = Plot(df, x=:Comb, y=:CO2, mode="markers", marker_size=8, text=:Make, group=:Fuel)
#p1 = PlotlyJS.scatter(df, x=:Comb, y=:CO2, mode="markers", marker_size=8, text=:Make, group=:Fuel)
c_r = load_object("D:/Master/S7/R-Julia/Projet/centrer_reduire.jld2")
SVC = MLJ.@load SVC pkg=LIBSVM
svc= load_object("D:/Master/S7/R-Julia/Projet/svm_classif.jld2")
KNNClassifier = MLJ.@load KNNClassifier
knn = JLD2.load_object("D:/Master/S7/R-Julia/Projet/knn_classif.jld2")
MyDecisionTree = MLJ.@load DecisionTreeClassifier pkg = DecisionTree
arbre_class = load_object("D:/Master/S7/R-Julia/Projet/arbre_classif.jld2")
MyDecisionTree2 = MLJ.@load DecisionTreeRegressor pkg = DecisionTree
arbre_predict = load_object("D:/Master/S7/R-Julia/Projet/arbre_predict.jdl2")
function f(x)
    if x == "E" return 0
    end
    return 1
end

function predire_knn(Marque, Modele, Class, Taille_moteur, Cylindres, Transmission,Fuel ,City,Hwy,Comb)
    c_r = load_object("D:/Master/S7/R-Julia/Projet/centrer_reduire.jld2")
    KNNClassifier = MLJ.@load KNNClassifier
    knn = JLD2.load_object("D:/Master/S7/R-Julia/Projet/knn_classif.jld2")
    fuel =  f(Fuel)  
    donnees = DataFrame(Engine=Taille_moteur, Cylinders = Cylindres,City=City, Hwy=Hwy, Comb=Comb )
    donnees = MLJ.transform(c_r, donnees)
    x1 = DataFrame(fuel=fuel)
    donnees=hcat(x1, donnees)
    s = string.(predict_mode(knn, donnees))
    s = s[1]
    return "L'intevalle est " * s[4:length(s)-1] * "] g/100km"
    
end

function predire_svm(Marque, Modele, Class, Taille_moteur, Cylindres, Transmission,Fuel ,City,Hwy,Comb)
    fuel =  f(Fuel)  
    donnees = DataFrame(Engine=Taille_moteur, Cylinders = Cylindres,City=City, Hwy=Hwy, Comb=Comb )
    donnees = MLJ.transform(c_r, donnees)
    x1 = DataFrame(fuel=fuel)
    donnees=hcat(x1, donnees)
    s = MLJ.predict(svc, donnees)
    s = string.(MLJ.predict(svc, donnees))
    s = s[1]
    return "L'intevalle est " * s[4:length(s)-1] * "] g/100km" 
end

function predire_arbre_int(Marque, Modele, Class, Taille_moteur, Cylindres, Transmission,Fuel ,City,Hwy,Comb)
    mpg = Comb *235.214583
    MyDecisionTree = MLJ.@load DecisionTreeClassifier pkg = DecisionTree
    arbre_class = load_object("D:/Master/S7/R-Julia/Projet/arbre_classif.jld2")
    donnees = DataFrame(Make=Marque, Model=Modele, Class=Class, Engine=Taille_moteur, Cylinders=Cylindres, Transmission=Transmission,Fuel=Fuel,City=City, Hwy=Hwy, Comb=Comb, mpg=mpg )
    s = string.(MLJ.predict_mode(arbre_class, donnees))
    s = s[1]
        return "L'intevalle est " * s[4:length(s)-1] * "] g/100km"
    
end

function predire_val_arbre(Marque, Modele, Class, Taille_moteur, Cylindres, Transmission,Fuel ,City,Hwy,Comb)
    mpg = Comb *235.214583
    MyDecisionTree2 = MLJ.@load DecisionTreeRegressor pkg = DecisionTree
    arbre_predict = load_object("D:/Master/S7/R-Julia/Projet/arbre_predict.jdl2")
    fuel =  f(Fuel)  
    donnees = DataFrame(Make=Marque, Model=Modele, Class=Class, Engine=Taille_moteur, Cylinders=Cylindres, Transmission=Transmission,Fuel=Fuel,City=City, Hwy=Hwy, Comb=Comb, mpg=mpg )
    x1 = DataFrame(fuel=fuel)
    donnees=hcat(x1, donnees)
    return MLJ.predict_mode(arbre_predict, donnees)[1]   
end


using Dash

app = dash()

app.layout = html_div([
    html_h4("CO2 = f(consommation mixte)"),
    dcc_graph(
        id = "example-graph-3",
        figure = p1,
    ),
    dcc_input(id = "Make", value = "Marque", type = "text"),
    dcc_input(id = "Model", value = "Modèle", type = "text"),
    dcc_input(id = "Class", value = "Classe", type = "text"),
    dcc_input(id = "Taille", value = "Taille du moteur", type = "text"),
    dcc_input(id = "Cylinders", value = "Nombre de cylindres", type = "text"),
    dcc_input(id = "Transmission", value = "Transmission", type = "text"),
    dcc_input(id = "Fuel", value = "Carburant", type = "text"),
    dcc_input(id = "City", value = "Consommation en ville", type = "text"),
    dcc_input(id = "Hwy", value = "Consommmation sur autoroute", type = "text"),
    dcc_input(id = "Comb", value = "Consommation mixte", type = "text"),
    html_tr((html_td("Prediction par knn : "), html_td(id = "pred_knn"))),
    html_tr((html_td("Prediction par svm : "), html_td(id = "pred_svm"))),
    html_tr((html_td("Prediction par arbre de classification : "), html_td(id = "pred_arbre_int"))),
    html_tr((html_td("Prediction par arbre de régression : "), html_td(id = "pred_arbre_val")))
    ])

callback!(app, Output("pred_knn", "children"), Output("pred_svm", "children"), Output("pred_arbre_int", "children"), Output("pred_arbre_val", "children"), 
    Input("Make", "value"),
    Input("Model", "value"),
    Input("Class", "value"),
    Input("Taille", "value"),
    Input("Cylinders", "value"),
    Input("Transmission", "value"),
    Input("Fuel", "value"),
    Input("City", "value"),
    Input("Hwy", "value"),
    Input("Comb", "value"),
) do Make, Model, Class, Taille, Cylinders, Transmission, Fuel, City, Hwy, Comb
if Taille == "Taille du moteur" || Cylinders == "Nombre de cylindres" || City == "Consommation en ville" || Hwy == "Consommation sur autoroute" || Comb == "Consommation mixte" 
    return ("Non","Non","Non","Non")
end
    if Taille == "" || Taille == nothing || Cylinders == "" || Cylinders == nothing || City == "" || City == nothing || Hwy == "" || Hwy == nothing || Comb == "" || Comb == nothing
        return ("Non","Non","Non","Non")
    end
    if Make == "" || Make == nothing || Model == "" || Model == nothing || Class == "" || Class == nothing || Transmission == "" || Transmission == nothing || Fuel == "" || Fuel == nothing
        return (
        predire_knn(Make, Model, Class, parse(Float64, Taille), parse(Float64, Cylinders), Transmission, Fuel, parse(Float64, City), parse(Float64,Hwy), parse(Float64, Comb)),
        predire_svm(Make, Model, Class, parse(Float64, Taille), parse(Float64, Cylinders), Transmission, Fuel, parse(Float64, City), parse(Float64,Hwy), parse(Float64, Comb)),
        "Non","Non")
    end
    p = predire_val_arbre(Make, Model, Class, parse(Float64, Taille), parse(Float64, Cylinders), Transmission, Fuel, parse(Float64, City), parse(Float64,Hwy), parse(Float64, Comb))[1]
    #scatter!([parse(Int64,Comb)],[trunc(Int32,p)], color="black", label = "nouveau", marker_size= 12)
    s = string.(p)
    return (
    predire_knn(Make, Model, Class, parse(Float64, Taille), parse(Float64, Cylinders), Transmission, Fuel, parse(Float64, City), parse(Float64,Hwy), parse(Float64, Comb)),
    predire_svm(Make, Model, Class, parse(Float64, Taille), parse(Float64, Cylinders), Transmission, Fuel, parse(Float64, City), parse(Float64,Hwy), parse(Float64, Comb)),
    predire_arbre_int(Make, Model, Class, parse(Float64, Taille), parse(Float64, Cylinders), Transmission, Fuel, parse(Float64, City), parse(Float64,Hwy), parse(Float64, Comb)),
    "La prédiction est " * s * " g/100km")
end

run_server(app, "0.0.0.0", debug=true)
