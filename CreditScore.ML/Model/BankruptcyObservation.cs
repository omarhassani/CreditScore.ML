using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Model
{
    public class BankruptcyObservation
    {
        /// <summary>
        /// Beneficio neto/Total Activo
        /// </summary>
        [LoadColumnAttribute(0)]
        public float V1 { get; set; }

        /// <summary>
        /// Activo Corriente/Pasivo Corriente
        /// </summary>
        [LoadColumnAttribute(1)]
        public float V2 { get; set; }

        /// <summary>
        /// (Activo Corriente - Pasivo Corriente)/Total Activo
        /// </summary>        
        [LoadColumnAttribute(2)]
        public float V3 { get; set; }

        /// <summary>
        /// Beneficio retenido / Total Activo = (Reservas año posterior – Reservas año actual) / Total Activo
        /// </summary>
        [LoadColumnAttribute(3)]
        public float V4 { get; set; }

        /// <summary>
        /// (Beneficio neto + Gastos financieros + Impuestos) / Total Activo
        /// </summary>
        [LoadColumnAttribute(4)]
        public float V5 { get; set; }

        /// <summary>
        /// Ventas / Total Activo
        /// </summary>
        [LoadColumnAttribute(5)]
        public float V6 { get; set; }


        /// <summary>
        /// (Tesorería + Deudores) / Pasivo Corriente
        /// </summary>
        [LoadColumnAttribute(6)]
        public float V7 { get; set; }

        /// <summary>
        /// Pasivo no corriente / Total Activo
        /// </summary>
        [LoadColumnAttribute(7)]
        public float V8 { get; set; }

        /// <summary>
        /// Activo corriente / Total Activo
        /// </summary>
        [LoadColumnAttribute(8)]
        public float V9 { get; set; }

        /// <summary>
        /// Beneficio neto / Patrimonio neto
        /// </summary>
        [LoadColumnAttribute(9)]
        public float V10 { get; set; }
        /// <summary>
        /// Total deudas / Total activo
        /// </summary>
        [LoadColumnAttribute(10)]
        public float V11 { get; set; }

        /// <summary>
        /// Tesorería / Total Activo
        /// </summary>
        [LoadColumnAttribute(11)]
        public float V12 { get; set; }

        /// <summary>
        /// (Beneficio neto + Dotación amortizaciones – Deudores año actual – Existencias año actual + Deudores año anterior + Existencias año anterior) / Total deudas
        /// </summary>
        [LoadColumnAttribute(12)]
        public float V13 { get; set; }


        /// <summary>
        /// CNAE
        /// </summary>
        [LoadColumnAttribute(13)]
        public string CNAE { get; set; }
    }
}
