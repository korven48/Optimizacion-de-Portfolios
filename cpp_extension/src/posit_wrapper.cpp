#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <universal/number/posit/posit.hpp>
#include <cmath>
#include <sstream>

namespace py = pybind11;
using namespace sw::universal;

// Wrapper de plantilla para diferentes configuraciones de posit
template<size_t nbits, size_t es>
class PositWrapper {
private:
    posit<nbits, es> value;

public:
    // Constructores
    PositWrapper() : value(0) {}
    PositWrapper(double d) : value(d) {}
    PositWrapper(int i) : value(i) {}
    PositWrapper(const posit<nbits, es>& p) : value(p) {}
    
    // Conversión a double
    double to_double() const {
        return double(value);
    }
    
    // Operadores aritméticos
    PositWrapper operator+(const PositWrapper& other) const {
        return PositWrapper(value + other.value);
    }
    
    PositWrapper operator-(const PositWrapper& other) const {
        return PositWrapper(value - other.value);
    }
    
    PositWrapper operator*(const PositWrapper& other) const {
        return PositWrapper(value * other.value);
    }
    
    PositWrapper operator/(const PositWrapper& other) const {
        return PositWrapper(value / other.value);
    }
    
    PositWrapper operator-() const {
        return PositWrapper(-value);
    }
    
    PositWrapper operator+() const {
        return *this;
    }
    
    // Operadores de comparación
    bool operator==(const PositWrapper& other) const {
        return value == other.value;
    }
    
    bool operator!=(const PositWrapper& other) const {
        return value != other.value;
    }
    
    bool operator<(const PositWrapper& other) const {
        return value < other.value;
    }
    
    bool operator>(const PositWrapper& other) const {
        return value > other.value;
    }
    
    bool operator<=(const PositWrapper& other) const {
        return value <= other.value;
    }
    
    bool operator>=(const PositWrapper& other) const {
        return value >= other.value;
    }
    
    // Representación en cadena
    std::string repr() const {
        std::ostringstream oss;
        oss << "Posit" << nbits << "<" << es << ">(" << double(value) << ")";
        return oss.str();
    }
    
    std::string str() const {
        std::ostringstream oss;
        oss << double(value);
        return oss.str();
    }
    
    // Funciones matemáticas
    PositWrapper sqrt_() const {
        return PositWrapper(sw::universal::sqrt(value));
    }
    
    PositWrapper exp_() const {
        return PositWrapper(sw::universal::exp(value));
    }
    
    PositWrapper log_() const {
        return PositWrapper(sw::universal::log(value));
    }
    
    PositWrapper abs_() const {
        return PositWrapper(sw::universal::abs(value));
    }
    
    PositWrapper pow_(const PositWrapper& exp) const {
        return PositWrapper(sw::universal::pow(value, exp.value));
    }
};

// Definición del módulo
PYBIND11_MODULE(posit, m) {
    m.doc() = "Wrapper aritmético de Posit para optimización de portafolios";
    
    // Posit8<2>
    py::class_<PositWrapper<8, 2>>(m, "Posit8")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<8, 2>::to_double)
        .def("__repr__", &PositWrapper<8, 2>::repr)
        .def("__str__", &PositWrapper<8, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<8, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<8, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<8, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<8, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<8, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<8, 2>::pow_);

    // Posit16<2>
    py::class_<PositWrapper<16, 2>>(m, "Posit16")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<16, 2>::to_double)
        .def("__repr__", &PositWrapper<16, 2>::repr)
        .def("__str__", &PositWrapper<16, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<16, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<16, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<16, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<16, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<16, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<16, 2>::pow_);

    // Posit32<2> - Configuración estándar
    py::class_<PositWrapper<32, 2>>(m, "Posit32")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<32, 2>::to_double)
        .def("__repr__", &PositWrapper<32, 2>::repr)
        .def("__str__", &PositWrapper<32, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<32, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<32, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<32, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<32, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<32, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<32, 2>::pow_);
    
    // Posit64<2> - Configuración de alta precisión
    py::class_<PositWrapper<64, 2>>(m, "Posit64")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def("__float__", &PositWrapper<64, 2>::to_double)
        .def("__repr__", &PositWrapper<64, 2>::repr)
        .def("__str__", &PositWrapper<64, 2>::str)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(+py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("sqrt", &PositWrapper<64, 2>::sqrt_, "Raíz cuadrada")
        .def("exp", &PositWrapper<64, 2>::exp_, "Exponencial")
        .def("log", &PositWrapper<64, 2>::log_, "Logaritmo natural")
        .def("abs", &PositWrapper<64, 2>::abs_, "Valor absoluto")
        .def("pow", &PositWrapper<64, 2>::pow_, "Potencia")
        .def("__pow__", &PositWrapper<64, 2>::pow_);
    
    // Funciones a nivel de módulo
    m.def("sqrt", [](const PositWrapper<8, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit8)");
    m.def("sqrt", [](const PositWrapper<16, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit16)");
    m.def("sqrt", [](const PositWrapper<32, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit32)");
    m.def("sqrt", [](const PositWrapper<64, 2>& p) { return p.sqrt_(); }, "Raíz cuadrada (Posit64)");
}
