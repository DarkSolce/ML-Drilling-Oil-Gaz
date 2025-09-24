if formation_metrics:
                metrics_df = pd.DataFrame(formation_metrics)
                
                # Display metrics table
                st.write("**Model Comparison Table:**")
                display_cols = ['Model', 'val_r2', 'val_rmse', 'val_mae']
                available_cols = [col for col in display_cols if col in metrics_df.columns]
                st.dataframe(metrics_df[available_cols].round(4))
                
                # Performance visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'val_r2' in metrics_df.columns:
                        fig_r2 = px.bar(
                            metrics_df, 
                            x='Model', 
                            y='val_r2',
                            title='Model R² Comparison',
                            color='val_r2',
                            color_continuous_scale='viridis'
                        )
                        fig_r2.update_layout(height=400)
                        st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    if 'val_rmse' in metrics_df.columns:
                        fig_rmse = px.bar(
                            metrics_df, 
                            x='Model', 
                            y='val_rmse',
                            title='Model RMSE Comparison',
                            color='val_rmse',
                            color_continuous_scale='viridis_r'
                        )
                        fig_rmse.update_layout(height=400)
                        st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Best model recommendation
                if 'val_r2' in metrics_df.columns:
                    best_model = metrics_df.loc[metrics_df['val_r2'].idxmax(), 'Model']
                    best_r2 = metrics_df['val_r2'].max()
                    
                    st.success(f"🏆 **Best Formation Pressure Model: {best_model}** (R² = {best_r2:.4f})")
        
        # Kick Detection Model Comparison
        if kick_models:
            st.subheader("🚨 Kick Detection Model Performance")
            
            # Get metrics for each kick model
            kick_metrics = []
            for model_type, model in kick_models.items():
                metrics_key = f'kick_metrics_{model_type}'
                if metrics_key in st.session_state:
                    metrics = st.session_state[metrics_key].copy()
                    metrics['Model'] = model_type
                    kick_metrics.append(metrics)
            
            if kick_metrics:
                kick_metrics_df = pd.DataFrame(kick_metrics)
                
                # Display metrics
                st.write("**Kick Detection Model Comparison:**")
                display_cols = ['Model', 'val_anomaly_rate']
                if 'val_accuracy' in kick_metrics_df.columns:
                    display_cols.extend(['val_accuracy', 'val_precision', 'val_recall'])
                
                available_cols = [col for col in display_cols if col in kick_metrics_df.columns]
                st.dataframe(kick_metrics_df[available_cols].round(4))
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'val_anomaly_rate' in kick_metrics_df.columns:
                        fig_anomaly = px.bar(
                            kick_metrics_df, 
                            x='Model', 
                            y='val_anomaly_rate',
                            title='Anomaly Detection Rate',
                            color='val_anomaly_rate',
                            color_continuous_scale='reds'
                        )
                        fig_anomaly.update_layout(height=400)
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                
                with col2:
                    if 'val_recall' in kick_metrics_df.columns:
                        fig_recall = px.bar(
                            kick_metrics_df, 
                            x='Model', 
                            y='val_recall',
                            title='Recall (Safety Critical)',
                            color='val_recall',
                            color_continuous_scale='greens'
                        )
                        fig_recall.update_layout(height=400)
                        st.plotly_chart(fig_recall, use_container_width=True)
                
                # Best kick model recommendation
                if 'val_recall' in kick_metrics_df.columns:
                    best_kick_model = kick_metrics_df.loc[kick_metrics_df['val_recall'].idxmax(), 'Model']
                    best_recall = kick_metrics_df['val_recall'].max()
                    st.success(f"🏆 **Best Kick Detection Model: {best_kick_model}** (Recall = {best_recall:.4f})")
                else:
                    best_kick_model = kick_metrics_df.loc[0, 'Model']  # First available
                    st.info(f"🏆 **Available Kick Detection Model: {best_kick_model}**")
        
        # Feature Importance Analysis
        st.subheader("🔍 Feature Importance Analysis")
        
        if formation_models:
            st.write("**Formation Pressure Models - Feature Importance:**")
            
            model_selector = st.selectbox(
                "Select Formation Model for Feature Analysis",
                options=list(formation_models.keys())
            )
            
            if model_selector:
                selected_model = formation_models[model_selector]
                
                try:
                    if hasattr(selected_model, 'get_feature_importance'):
                        importance = selected_model.get_feature_importance(top_k=10)
                        
                        if importance:
                            # Create feature importance dataframe
                            importance_df = pd.DataFrame([
                                {'Feature': feature, 'Importance': imp}
                                for feature, imp in importance.items()
                            ])
                            
                            # Plot feature importance
                            fig_importance = px.bar(
                                importance_df, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title=f'Feature Importance - {model_selector}',
                                color='Importance',
                                color_continuous_scale='blues'
                            )
                            fig_importance.update_layout(height=500)
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Display importance table
                            st.dataframe(importance_df.round(4))
                        else:
                            st.info("Feature importance not available for this model type")
                    else:
                        st.info("Feature importance not available for this model type")
                
                except Exception as e:
                    st.error(f"Error analyzing feature importance: {str(e)}")
        
        # Model Export Section
        st.subheader("💾 Model Export & Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Trained Models:**")
            
            if formation_models:
                for model_name, model in formation_models.items():
                    if st.button(f"📁 Export {model_name} Formation Model"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filepath = f"formation_{model_name}_{timestamp}.pkl"
                            saved_path = model.save_model(filepath)
                            st.success(f"✅ Model saved: {saved_path}")
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
            
            if kick_models:
                for model_name, model in kick_models.items():
                    if st.button(f"🚨 Export {model_name} Kick Model"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filepath = f"kick_{model_name}_{timestamp}.pkl"
                            saved_path = model.save_model(filepath)
                            st.success(f"✅ Model saved: {saved_path}")
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
        
        with col2:
            st.write("**Model Management:**")
            
            if st.button("🗑️ Clear All Models"):
                # Clear all model-related session state
                keys_to_remove = [key for key in st.session_state.keys() 
                                if key.startswith(('formation_model_', 'kick_model_', 
                                                 'formation_metrics_', 'kick_metrics_'))]
                
                for key in keys_to_remove:
                    del st.session_state[key]
                
                st.success("✅ All models cleared from memory")
                st.rerun()
            
            if st.button("📊 Generate Performance Report"):
                # Generate comprehensive performance report
                report = self.generate_performance_report(formation_models, kick_models)
                
                st.download_button(
                    label="📄 Download Performance Report",
                    data=report,
                    file_name=f"drilling_ml_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    def generate_performance_report(self, formation_models, kick_models):
        """Generate comprehensive performance report"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DRILLING OPERATIONS ML PERFORMANCE REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        
        # Formation Pressure Models
        if formation_models:
            report_lines.append("\n📊 FORMATION PRESSURE MODELS")
            report_lines.append("-" * 40)
            
            for model_type, model in formation_models.items():
                metrics_key = f'formation_metrics_{model_type}'
                if metrics_key in st.session_state:
                    metrics = st.session_state[metrics_key]
                    
                    report_lines.append(f"\n{model_type.upper()} MODEL:")
                    report_lines.append(f"  R² Score: {metrics.get('val_r2', 'N/A'):.4f}")
                    report_lines.append(f"  RMSE: {metrics.get('val_rmse', 'N/A'):.2f}")
                    report_lines.append(f"  MAE: {metrics.get('val_mae', 'N/A'):.2f}")
                    
                    if 'explained_variance_ratio' in metrics:
                        report_lines.append(f"  Explained Variance: {metrics['explained_variance_ratio']:.4f}")
        
        # Kick Detection Models
        if kick_models:
            report_lines.append("\n\n🚨 KICK DETECTION MODELS")
            report_lines.append("-" * 40)
            
            for model_type, model in kick_models.items():
                metrics_key = f'kick_metrics_{model_type}'
                if metrics_key in st.session_state:
                    metrics = st.session_state[metrics_key]
                    
                    report_lines.append(f"\n{model_type.upper()} MODEL:")
                    report_lines.append(f"  Anomaly Rate: {metrics.get('val_anomaly_rate', 'N/A'):.4f}")
                    
                    if 'val_accuracy' in metrics:
                        report_lines.append(f"  Accuracy: {metrics['val_accuracy']:.4f}")
                        report_lines.append(f"  Precision: {metrics.get('val_precision', 'N/A'):.4f}")
                        report_lines.append(f"  Recall: {metrics.get('val_recall', 'N/A'):.4f}")
        
        # Recommendations
        report_lines.append("\n\n💡 RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if formation_models:
            # Find best formation model
            best_r2 = 0
            best_formation_model = None
            
            for model_type in formation_models.keys():
                metrics_key = f'formation_metrics_{model_type}'
                if metrics_key in st.session_state:
                    r2 = st.session_state[metrics_key].get('val_r2', 0)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_formation_model = model_type
            
            if best_formation_model:
                report_lines.append(f"🏆 Best Formation Pressure Model: {best_formation_model} (R² = {best_r2:.4f})")
        
        if kick_models:
            # Find best kick model (prefer high recall for safety)
            best_recall = 0
            best_kick_model = None
            
            for model_type in kick_models.keys():
                metrics_key = f'kick_metrics_{model_type}'
                if metrics_key in st.session_state:
                    recall = st.session_state[metrics_key].get('val_recall', 0)
                    if recall > best_recall:
                        best_recall = recall
                        best_kick_model = model_type
            
            if best_kick_model:
                report_lines.append(f"🚨 Best Kick Detection Model: {best_kick_model} (Recall = {best_recall:.4f})")
        
        # Usage Guidelines
        report_lines.append("\n\n📋 USAGE GUIDELINES")
        report_lines.append("-" * 40)
        report_lines.append("• Formation Pressure: Use for mud weight optimization and well planning")
        report_lines.append("• Kick Detection: Critical for safety - prioritize high recall over precision")
        report_lines.append("• Real-time Monitoring: Update predictions every 30-60 seconds")
        report_lines.append("• Model Retraining: Recommended every 30 days with new data")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def run(self):
        """Main dashboard runner"""
        
        # Render sidebar and get page selection
        page, formation_model_type, kick_model_type, simulate_realtime = self.render_sidebar()
        
        # Render selected page
        if page == "Overview":
            self.render_overview_page()
        
        elif page == "Formation Pressure":
            self.render_formation_pressure_page(formation_model_type)
        
        elif page == "Kick Detection":
            self.render_kick_detection_page(kick_model_type)
        
        elif page == "Real-time Monitoring":
            self.render_realtime_monitoring_page(simulate_realtime)
        
        elif page == "Model Performance":
            self.render_model_performance_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
        🛢️ Drilling Operations ML Dashboard | Powered by Streamlit & Python ML
        </div>
        """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(90deg, #1f2937 0%, #374151 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .drilling-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .emergency-alert {
        background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    try:
        dashboard = DrillingDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"❌ Dashboard Error: {str(e)}")
        st.info("Please check your data files and model configurations.")
        
        # Show debug information in expander
        with st.expander("🔍 Debug Information"):
            import traceback
            st.code(traceback.format_exc())
 Train model
                    metrics = model.train(X, y, validation_split=validation_split if show_advanced else 0.2)
                    
                    # Store model in session state
                    st.session_state[f'formation_model_{model_type}'] = model
                    st.session_state[f'formation_metrics_{model_type}'] = metrics
                    
                    st.success(f"✅ {model_type} model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    return
        
        # Display model metrics
        if f'formation_metrics_{model_type}' in st.session_state:
            metrics = st.session_state[f'formation_metrics_{model_type}']
            
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{metrics['val_r2']:.4f}")
            
            with col2:
                st.metric("RMSE", f"{metrics['val_rmse']:.2f}")
            
            with col3:
                st.metric("MAE", f"{metrics['val_mae']:.2f}")
            
            with col4:
                if 'explained_variance_ratio' in metrics:
                    st.metric("Explained Variance", f"{metrics['explained_variance_ratio']:.3f}")
                else:
                    st.metric("Training Status", "✅ Complete")
        
        # Prediction section
        st.subheader("🔮 Make Predictions")
        
        if f'formation_model_{model_type}' in st.session_state:
            model = st.session_state[f'formation_model_{model_type}']
            
            # Manual input for prediction
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Input Parameters:**")
                well_depth = st.number_input("Well Depth (ft)", value=5000, min_value=0)
                wob = st.number_input("Weight on Bit (klbs)", value=25, min_value=0)
                rop = st.number_input("Rate of Penetration (ft/hr)", value=15, min_value=0)
                torque = st.number_input("Bit Torque (klb-ft)", value=120, min_value=0)
            
            with col2:
                wb_pressure = st.number_input("Wellbore Pressure (psi)", value=2000, min_value=0)
                hook_load = st.number_input("Hook Load (klbs)", value=150, min_value=0)
                dp_pressure = st.number_input("Differential Pressure (psi)", value=180, min_value=0)
            
            if st.button("🎯 Predict Formation Pressure"):
                try:
                    # Create input dataframe
                    input_data = pd.DataFrame({
                        'WellDepth': [well_depth],
                        'WoBit': [wob],
                        'RoPen': [rop],
                        'BTBR': [torque],
                        'WBoPress': [wb_pressure],
                        'HLoad': [hook_load],
                        'DPPress': [dp_pressure]
                    })
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    
                    # Display result
                    st.success(f"🎯 **Predicted Formation Pressure: {prediction:.2f} psi**")
                    
                    # Additional analysis
                    pressure_gradient = prediction / well_depth
                    normal_gradient = 0.433
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Pressure Gradient", f"{pressure_gradient:.3f} psi/ft")
                    
                    with col2:
                        gradient_ratio = pressure_gradient / normal_gradient
                        st.metric("Gradient Ratio", f"{gradient_ratio:.2f}")
                    
                    with col3:
                        if gradient_ratio > 1.2:
                            st.warning("⚠️ High Pressure Zone")
                        elif gradient_ratio < 0.8:
                            st.info("ℹ️ Low Pressure Zone")
                        else:
                            st.success("✅ Normal Pressure")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
        # Historical predictions visualization
        if f'formation_model_{model_type}' in st.session_state:
            model = st.session_state[f'formation_model_{model_type}']
            
            st.subheader("📈 Historical Data Visualization")
            
            # Get recent data for plotting
            recent_data = formation_data.tail(500)
            
            if len(recent_data) > 0 and 'FPress' in recent_data.columns:
                # Make predictions on recent data
                feature_cols = [col for col in recent_data.columns if col != 'FPress']
                X_recent = recent_data[feature_cols]
                
                try:
                    y_pred = model.predict(X_recent)
                    y_actual = recent_data['FPress'].values
                    
                    # Create comparison plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(y_actual))),
                        y=y_actual,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(y_pred))),
                        y=y_pred,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Formation Pressure: Actual vs Predicted',
                        xaxis_title='Data Points',
                        yaxis_title='Formation Pressure (psi)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residuals plot
                    residuals = y_actual - y_pred
                    
                    fig_residuals = px.scatter(
                        x=y_pred, 
                        y=residuals,
                        title='Residuals Plot',
                        labels={'x': 'Predicted Pressure (psi)', 'y': 'Residuals (psi)'}
                    )
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig_residuals, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Visualization error: {str(e)}")
    
    def render_kick_detection_page(self, model_type):
        """Render kick detection page"""
        st.title("⚠️ Kick Detection System")
        
        if not self.load_data():
            return
        
        kick_data = st.session_state.kick_data
        
        # Model training section
        st.subheader("🚨 Anomaly Detection Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            train_kick_model = st.button("🚀 Train Kick Detection Model", type="primary")
        
        with col2:
            show_kick_advanced = st.checkbox("Show Advanced Options", key="kick_advanced")
        
        if show_kick_advanced:
            st.subheader("⚙️ Detection Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                detection_threshold = st.slider("Detection Threshold (%)", 90.0, 99.9, 99.0)
            
            with col2:
                if model_type == "PCA":
                    variance_threshold = st.slider("PCA Variance", 0.8, 0.99, 0.9)
                contamination = st.slider("Expected Anomaly Rate", 0.01, 0.2, 0.05)
            
            with col3:
                use_synthetic_kicks = st.checkbox("Add Synthetic Kicks", value=True)
        
        # Train kick detection model
        if train_kick_model or f'kick_model_{model_type}' not in st.session_state:
            with st.spinner(f"Training {model_type} kick detection model..."):
                try:
                    # Prepare data
                    preprocessor = DataPreprocessor()
                    processed_data = preprocessor.prepare_kick_detection_data(kick_data)
                    
                    # Select features
                    feature_cols = [col for col in processed_data.columns if col != 'ActiveGL']
                    X = processed_data[feature_cols]
                    
                    # Create labels (synthetic or based on pit volume changes)
                    if use_synthetic_kicks if show_kick_advanced else True:
                        # Create synthetic kick labels based on pit volume anomalies
                        pit_volume_changes = processed_data['ActiveGL'].diff().abs()
                        threshold_99 = pit_volume_changes.quantile(0.99)
                        y = (pit_volume_changes > threshold_99).astype(int)
                    else:
                        y = None
                    
                    # Initialize model
                    if model_type == "PCA":
                        model = PCAKickDetection(
                            variance_threshold=variance_threshold if show_kick_advanced else 0.9
                        )
                    elif model_type == "Ensemble":
                        model = EnsembleKickDetection(['pca', 'isolation_forest'])
                    
                    # Train model
                    model_params = {}
                    if show_kick_advanced:
                        model_params['detection_threshold'] = detection_threshold
                        if hasattr(model, 'contamination'):
                            model_params['contamination'] = contamination
                    
                    metrics = model.train(X, y, **model_params)
                    
                    # Store model
                    st.session_state[f'kick_model_{model_type}'] = model
                    st.session_state[f'kick_metrics_{model_type}'] = metrics
                    
                    st.success(f"✅ {model_type} kick detection model trained!")
                    
                except Exception as e:
                    st.error(f"❌ Error training kick detection model: {str(e)}")
                    return
        
        # Display kick detection metrics
        if f'kick_metrics_{model_type}' in st.session_state:
            metrics = st.session_state[f'kick_metrics_{model_type}']
            
            st.subheader("📊 Detection Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                anomaly_rate = metrics.get('val_anomaly_rate', 0)
                st.metric("Anomaly Detection Rate", f"{anomaly_rate:.3f}")
            
            with col2:
                if 'spe_threshold' in metrics:
                    st.metric("SPE Threshold", f"{metrics['spe_threshold']:.2f}")
                elif 'contamination' in metrics:
                    st.metric("Contamination", f"{metrics['contamination']:.3f}")
            
            with col3:
                if 'explained_variance_ratio' in metrics:
                    st.metric("Explained Variance", f"{metrics['explained_variance_ratio']:.3f}")
                else:
                    st.metric("Detection Method", model_type)
            
            with col4:
                if 'val_accuracy' in metrics:
                    st.metric("Accuracy", f"{metrics['val_accuracy']:.3f}")
                else:
                    st.metric("Status", "✅ Ready")
        
        # Real-time kick detection
        st.subheader("🔍 Real-time Kick Monitoring")
        
        if f'kick_model_{model_type}' in st.session_state:
            model = st.session_state[f'kick_model_{model_type}']
            
            # Current drilling parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Drilling Parameters:**")
                current_flow_in = st.number_input("Flow In (gpm)", value=300, min_value=0)
                current_flow_out = st.number_input("Flow Out (gpm)", value=305, min_value=0)
                current_pit_volume = st.number_input("Active Pit Volume (bbl)", value=100, min_value=0)
                standpipe_pressure = st.number_input("Standpipe Pressure (psi)", value=2000, min_value=0)
            
            with col2:
                st.write("**Additional Parameters:**")
                mud_flow_return = st.number_input("Mud Return Flow (gpm)", value=295, min_value=0)
                hook_load = st.number_input("Hook Load (klbs)", value=150, min_value=0)
                block_speed = st.number_input("Block Speed (ft/min)", value=50, min_value=0)
            
            if st.button("🚨 Check for Kick"):
                try:
                    # Create input data
                    input_data = pd.DataFrame({
                        'FIn': [current_flow_in],
                        'FOut': [current_flow_out],
                        'ActiveGL': [current_pit_volume],
                        'WBoPress': [standpipe_pressure],
                        'MRFlow': [mud_flow_return],
                        'HLoad': [hook_load],
                        'SMSpeed': [block_speed]
                    })
                    
                    # Add any missing columns with default values
                    for col in model.feature_columns:
                        if col not in input_data.columns:
                            input_data[col] = 0
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    
                    # Get anomaly score if available
                    if hasattr(model, 'get_anomaly_scores'):
                        anomaly_score = model.get_anomaly_scores(input_data)[0]
                    else:
                        anomaly_score = None
                    
                    # Display results
                    if prediction == 1:
                        st.error("🚨 **KICK DETECTED!**")
                        st.error("⚠️ **IMMEDIATE ACTION REQUIRED**")
                        
                        # Emergency recommendations
                        st.subheader("🆘 Emergency Procedures:")
                        emergency_procedures = [
                            "1. **STOP DRILLING** - Cease all drilling operations immediately",
                            "2. **CLOSE BOP** - Activate blowout preventer if necessary", 
                            "3. **MONITOR PRESSURES** - Watch standpipe and casing pressures",
                            "4. **CIRCULATE** - Begin proper kick circulation procedures",
                            "5. **WEIGH UP MUD** - Increase mud weight as calculated",
                            "6. **NOTIFY SUPERVISOR** - Alert drilling supervisor and company man"
                        ]
                        
                        for procedure in emergency_procedures:
                            st.write(procedure)
                        
                    else:
                        st.success("✅ **Normal Drilling Conditions**")
                        st.info("Continue normal drilling operations")
                    
                    # Additional information
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        flow_balance = current_flow_out - current_flow_in
                        st.metric("Flow Balance", f"{flow_balance} gpm")
                        
                        if abs(flow_balance) > 10:
                            st.warning("⚠️ Flow imbalance detected")
                    
                    with col2:
                        if anomaly_score is not None:
                            st.metric("Anomaly Score", f"{anomaly_score:.3f}")
                    
                    with col3:
                        confidence = "High" if prediction == 1 else "Medium"
                        st.metric("Confidence", confidence)
                
                except Exception as e:
                    st.error(f"❌ Detection error: {str(e)}")
        
        # Historical kick analysis
        if f'kick_model_{model_type}' in st.session_state:
            model = st.session_state[f'kick_model_{model_type}']
            
            st.subheader("📊 Historical Anomaly Analysis")
            
            # Analyze recent data
            recent_data = kick_data.tail(1000)
            
            if len(recent_data) > 0:
                try:
                    # Prepare features
                    feature_cols = [col for col in recent_data.columns if col in model.feature_columns]
                    X_recent = recent_data[feature_cols]
                    
                    # Fill missing columns
                    for col in model.feature_columns:
                        if col not in X_recent.columns:
                            X_recent[col] = 0
                    
                    # Get predictions and scores
                    predictions = model.predict(X_recent)
                    
                    if hasattr(model, 'get_anomaly_scores'):
                        anomaly_scores = model.get_anomaly_scores(X_recent)
                    else:
                        anomaly_scores = predictions.astype(float)
                    
                    # Create timeline plot
                    fig = go.Figure()
                    
                    # Plot anomaly scores
                    fig.add_trace(go.Scatter(
                        x=list(range(len(anomaly_scores))),
                        y=anomaly_scores,
                        mode='lines',
                        name='Anomaly Score',
                        line=dict(color='blue')
                    ))
                    
                    # Highlight detected kicks
                    kick_indices = np.where(predictions == 1)[0]
                    if len(kick_indices) > 0:
                        fig.add_trace(go.Scatter(
                            x=kick_indices,
                            y=anomaly_scores[kick_indices],
                            mode='markers',
                            name='Detected Kicks',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig.update_layout(
                        title='Kick Detection Timeline',
                        xaxis_title='Time (Data Points)',
                        yaxis_title='Anomaly Score',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_kicks = np.sum(predictions)
                        st.metric("Total Kicks Detected", int(total_kicks))
                    
                    with col2:
                        kick_rate = total_kicks / len(predictions) * 100
                        st.metric("Kick Rate", f"{kick_rate:.2f}%")
                    
                    with col3:
                        max_anomaly = np.max(anomaly_scores)
                        st.metric("Max Anomaly Score", f"{max_anomaly:.3f}")
                
                except Exception as e:
                    st.error(f"❌ Historical analysis error: {str(e)}")
    
    def render_realtime_monitoring_page(self, simulate_realtime):
        """Render real-time monitoring page"""
        st.title("📡 Real-time Drilling Monitoring")
        
        if not self.load_data():
            return
        
        # Check if models are available
        has_formation_model = any(key.startswith('formation_model_') for key in st.session_state.keys())
        has_kick_model = any(key.startswith('kick_model_') for key in st.session_state.keys())
        
        if not (has_formation_model or has_kick_model):
            st.warning("⚠️ Please train at least one model first (Formation Pressure or Kick Detection)")
            return
        
        # Real-time simulation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if simulate_realtime:
                st.success("🟢 Real-time Mode: ACTIVE")
            else:
                st.info("⚪ Real-time Mode: INACTIVE")
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=simulate_realtime)
        
        with col3:
            manual_refresh = st.button("🔄 Refresh Now")
        
        # Create placeholder for real-time data
        if simulate_realtime or manual_refresh:
            
            # Simulate real-time drilling data
            current_time = datetime.now()
            
            # Generate synthetic current drilling parameters
            np.random.seed(int(current_time.timestamp()) % 1000)
            
            current_data = {
                'timestamp': current_time,
                'well_depth': 5000 + np.random.normal(0, 100),
                'wob': 25 + np.random.normal(0, 5),
                'rop': 15 + np.random.normal(0, 3),
                'torque': 120 + np.random.normal(0, 15),
                'standpipe_pressure': 2000 + np.random.normal(0, 100),
                'flow_in': 300 + np.random.normal(0, 20),
                'flow_out': 302 + np.random.normal(0, 20),
                'active_pit_volume': 100 + np.random.normal(0, 5),
                'hook_load': 150 + np.random.normal(0, 10)
            }
            
            # Display current status
            st.subheader(f"📊 Current Status - {current_time.strftime('%H:%M:%S')}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Well Depth", f"{current_data['well_depth']:.0f} ft")
                st.metric("WOB", f"{current_data['wob']:.1f} klbs")
            
            with col2:
                st.metric("ROP", f"{current_data['rop']:.1f} ft/hr")
                st.metric("Torque", f"{current_data['torque']:.0f} klb-ft")
            
            with col3:
                st.metric("Standpipe P.", f"{current_data['standpipe_pressure']:.0f} psi")
                st.metric("Flow In", f"{current_data['flow_in']:.0f} gpm")
            
            with col4:
                st.metric("Flow Out", f"{current_data['flow_out']:.0f} gpm")
                flow_balance = current_data['flow_out'] - current_data['flow_in']
                st.metric("Flow Balance", f"{flow_balance:.1f} gpm")
            
            with col5:
                st.metric("Pit Volume", f"{current_data['active_pit_volume']:.1f} bbl")
                st.metric("Hook Load", f"{current_data['hook_load']:.0f} klbs")
            
            # Formation pressure prediction
            if has_formation_model:
                st.subheader("🎯 Formation Pressure Prediction")
                
                # Get the first available formation model
                formation_model_key = next(key for key in st.session_state.keys() if key.startswith('formation_model_'))
                formation_model = st.session_state[formation_model_key]
                
                try:
                    # Create input for prediction
                    formation_input = pd.DataFrame({
                        'WellDepth': [current_data['well_depth']],
                        'WoBit': [current_data['wob']],
                        'RoPen': [current_data['rop']],
                        'BTBR': [current_data['torque']],
                        'WBoPress': [current_data['standpipe_pressure']],
                        'HLoad': [current_data['hook_load']],
                        'DPPress': [180]  # Default value
                    })
                    
                    predicted_pressure = formation_model.predict(formation_input)[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Formation Pressure", f"{predicted_pressure:.0f} psi")
                    
                    with col2:
                        pressure_gradient = predicted_pressure / current_data['well_depth']
                        st.metric("Pressure Gradient", f"{pressure_gradient:.3f} psi/ft")
                    
                    with col3:
                        mud_weight_equiv = predicted_pressure * 0.052
                        required_mud_weight = mud_weight_equiv + 0.5  # Safety margin
                        st.metric("Required Mud Weight", f"{required_mud_weight:.1f} ppg")
                
                except Exception as e:
                    st.error(f"Formation pressure prediction error: {str(e)}")
            
            # Kick detection
            if has_kick_model:
                st.subheader("🚨 Kick Detection Status")
                
                # Get the first available kick model
                kick_model_key = next(key for key in st.session_state.keys() if key.startswith('kick_model_'))
                kick_model = st.session_state[kick_model_key]
                
                try:
                    # Create input for kick detection
                    kick_input = pd.DataFrame({
                        'FIn': [current_data['flow_in']],
                        'FOut': [current_data['flow_out']],
                        'ActiveGL': [current_data['active_pit_volume']],
                        'WBoPress': [current_data['standpipe_pressure']],
                        'HLoad': [current_data['hook_load']],
                        'MRFlow': [current_data['flow_out'] - 5],  # Approximate
                        'SMSpeed': [50]  # Default
                    })
                    
                    # Add missing features with defaults
                    for col in kick_model.feature_columns:
                        if col not in kick_input.columns:
                            kick_input[col] = 0
                    
                    kick_prediction = kick_model.predict(kick_input)[0]
                    
                    if hasattr(kick_model, 'get_anomaly_scores'):
                        anomaly_score = kick_model.get_anomaly_scores(kick_input)[0]
                    else:
                        anomaly_score = float(kick_prediction)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if kick_prediction == 1:
                            st.error("🚨 KICK DETECTED!")
                        else:
                            st.success("✅ Normal Conditions")
                    
                    with col2:
                        st.metric("Anomaly Score", f"{anomaly_score:.3f}")
                    
                    with col3:
                        if abs(flow_balance) > 10:
                            st.warning("⚠️ Flow Imbalance")
                        else:
                            st.info("Flow Balanced")
                
                except Exception as e:
                    st.error(f"Kick detection error: {str(e)}")
            
            # Historical trend (last 20 points)
            st.subheader("📈 Recent Trends")
            
            # Generate historical data points
            historical_times = [current_time - timedelta(minutes=i) for i in range(19, -1, -1)]
            historical_data = []
            
            for i, time_point in enumerate(historical_times):
                np.random.seed(int(time_point.timestamp()) % 1000)
                point = {
                    'time': time_point,
                    'rop': 15 + np.random.normal(0, 2),
                    'wob': 25 + np.random.normal(0, 3),
                    'flow_balance': np.random.normal(2, 5)
                }
                historical_data.append(point)
            
            # Create trend plots
            historical_df = pd.DataFrame(historical_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rop = px.line(
                    historical_df, 
                    x='time', 
                    y='rop',
                    title='Rate of Penetration Trend',
                    labels={'rop': 'ROP (ft/hr)', 'time': 'Time'}
                )
                fig_rop.update_layout(height=300)
                st.plotly_chart(fig_rop, use_container_width=True)
            
            with col2:
                fig_flow = px.line(
                    historical_df, 
                    x='time', 
                    y='flow_balance',
                    title='Flow Balance Trend',
                    labels={'flow_balance': 'Flow Balance (gpm)', 'time': 'Time'}
                )
                fig_flow.add_hline(y=0, line_dash="dash", line_color="red")
                fig_flow.update_layout(height=300)
                st.plotly_chart(fig_flow, use_container_width=True)
            
        # Auto-refresh logic
        if simulate_realtime and auto_refresh:
            update_interval = getattr(st.session_state, 'update_interval', 5)
            time.sleep(1)  # Small delay
            st.rerun()
    
    def render_model_performance_page(self):
        """Render model performance comparison page"""
        st.title("🏆 Model Performance Analysis")
        
        # Check available models
        formation_models = {key.split('_')[-1]: st.session_state[key] 
                          for key in st.session_state.keys() 
                          if key.startswith('formation_model_')}
        
        kick_models = {key.split('_')[-1]: st.session_state[key] 
                     for key in st.session_state.keys() 
                     if key.startswith('kick_model_')}
        
        if not formation_models and not kick_models:
            st.warning("⚠️ No trained models found. Please train models first.")
            return
        
        # Formation Pressure Model Comparison
        if formation_models:
            st.subheader("🎯 Formation Pressure Model Performance")
            
            # Get metrics for each model
            formation_metrics = []
            for model_type, model in formation_models.items():
                metrics_key = f'formation_metrics_{model_type}'
                if metrics_key in st.session_state:
                    metrics = st.session_state[metrics_key].copy()
                    metrics['Model'] = model_type
                    formation_metrics.append(metrics)
            
            if formation_metrics:
                metrics_df = pd.DataFrame(formation_metrics)



 #"""
Streamlit Dashboard for Drilling Operations ML
Interactive dashboard for monitoring and predicting drilling parameters
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from models.formation_pressure import PCRFormationPressure, XGBoostFormationPressure, EnsembleFormationPressure
from models.kick_detection import PCAKickDetection, EnsembleKickDetection
from utils.config import config

# Dashboard configuration
st.set_page_config(
    page_title="Drilling Operations ML Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DrillingDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_loader = None
        self.models = {}
        self.data = {}
        
    def load_data(self):
        """Load and cache data"""
        if 'data_loader' not in st.session_state:
            st.session_state.data_loader = DataLoader()
            
            # Load both datasets
            try:
                formation_data = st.session_state.data_loader.load_formation_data()
                kick_data = st.session_state.data_loader.load_kick_data()
                
                st.session_state.formation_data = formation_data
                st.session_state.kick_data = kick_data
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False
        
        return True
    
    def load_models(self):
        """Load and cache models"""
        if 'models_loaded' not in st.session_state:
            st.session_state.models = {}
            st.session_state.models_loaded = False
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("🛢️ Drilling ML Dashboard")
        st.sidebar.markdown("---")
        
        # Page selection
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Formation Pressure", "Kick Detection", "Real-time Monitoring", "Model Performance"]
        )
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data"):
            self.clear_cache()
            st.rerun()
        
        # Model controls
        st.sidebar.markdown("### Model Controls")
        
        # Formation pressure model selection
        formation_model_type = st.sidebar.selectbox(
            "Formation Pressure Model",
            ["PCR", "XGBoost", "Ensemble"]
        )
        
        # Kick detection model selection
        kick_model_type = st.sidebar.selectbox(
            "Kick Detection Model",
            ["PCA", "Ensemble"]
        )
        
        # Real-time simulation
        st.sidebar.markdown("### Real-time Simulation")
        simulate_realtime = st.sidebar.checkbox("Enable Real-time Simulation")
        
        if simulate_realtime:
            update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 3)
            st.session_state.update_interval = update_interval
        
        return page, formation_model_type, kick_model_type, simulate_realtime
    
    def clear_cache(self):
        """Clear cached data"""
        for key in list(st.session_state.keys()):
            if key.startswith(('data_', 'models_', 'formation_', 'kick_')):
                del st.session_state[key]
    
    def render_overview_page(self):
        """Render overview page"""
        st.title("📊 Drilling Operations Overview")
        
        if not self.load_data():
            return
        
        # Get data
        formation_data = st.session_state.formation_data
        kick_data = st.session_state.kick_data
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Formation Data Points",
                f"{len(formation_data):,}",
                delta=f"{len(formation_data) - 1000:,}" if len(formation_data) > 1000 else None
            )
        
        with col2:
            st.metric(
                "Kick Data Points",
                f"{len(kick_data):,}",
                delta=f"{len(kick_data) - 1000:,}" if len(kick_data) > 1000 else None
            )
        
        with col3:
            avg_depth = formation_data['WellDepth'].mean() if 'WellDepth' in formation_data.columns else 0
            st.metric(
                "Average Well Depth",
                f"{avg_depth:.0f} ft",
                delta="↗️" if avg_depth > 5000 else "↘️"
            )
        
        with col4:
            avg_pressure = formation_data['FPress'].mean() if 'FPress' in formation_data.columns else 0
            st.metric(
                "Average Formation Pressure",
                f"{avg_pressure:.0f} psi",
                delta="↗️" if avg_pressure > 2000 else "↘️"
            )
        
        # Data quality overview
        st.subheader("📈 Data Quality Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Formation data quality
            st.write("**Formation Data Quality**")
            missing_data = formation_data.isnull().sum().sum()
            total_data = formation_data.size
            quality_score = (1 - missing_data / total_data) * 100
            
            st.progress(quality_score / 100)
            st.write(f"Quality Score: {quality_score:.1f}%")
            st.write(f"Missing Values: {missing_data:,}")
        
        with col2:
            # Kick data quality
            st.write("**Kick Detection Data Quality**")
            missing_data = kick_data.isnull().sum().sum()
            total_data = kick_data.size
            quality_score = (1 - missing_data / total_data) * 100
            
            st.progress(quality_score / 100)
            st.write(f"Quality Score: {quality_score:.1f}%")
            st.write(f"Missing Values: {missing_data:,}")
        
        # Recent trends
        st.subheader("📊 Recent Drilling Trends")
        
        # Formation pressure trend
        if 'FPress' in formation_data.columns and 'WellDepth' in formation_data.columns:
            fig = px.scatter(
                formation_data.tail(500), 
                x='WellDepth', 
                y='FPress',
                title='Formation Pressure vs Depth (Recent 500 points)',
                labels={'WellDepth': 'Well Depth (ft)', 'FPress': 'Formation Pressure (psi)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data distribution
        st.subheader("📊 Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RoPen' in formation_data.columns:
                fig = px.histogram(
                    formation_data, 
                    x='RoPen',
                    title='Rate of Penetration Distribution',
                    labels={'RoPen': 'ROP (ft/hr)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'ActiveGL' in kick_data.columns:
                fig = px.histogram(
                    kick_data, 
                    x='ActiveGL',
                    title='Active Pit Volume Distribution',
                    labels={'ActiveGL': 'Active Volume (bbl)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_formation_pressure_page(self, model_type):
        """Render formation pressure prediction page"""
        st.title("🎯 Formation Pressure Prediction")
        
        if not self.load_data():
            return
        
        formation_data = st.session_state.formation_data
        
        # Model training section
        st.subheader("🤖 Model Training & Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            train_model = st.button("🚀 Train Formation Pressure Model", type="primary")
        
        with col2:
            show_advanced = st.checkbox("Show Advanced Options")
        
        if show_advanced:
            st.subheader("⚙️ Advanced Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            
            with col2:
                if model_type == "PCR":
                    n_components = st.slider("PCA Components", 2, 15, 4)
                elif model_type == "XGBoost":
                    n_estimators = st.slider("N Estimators", 50, 300, 100)
            
            with col3:
                apply_smoothing = st.checkbox("Apply Data Smoothing", value=True)
        
        # Train model
        if train_model or f'formation_model_{model_type}' not in st.session_state:
            with st.spinner(f"Training {model_type} model..."):
                try:
                    # Prepare data
                    preprocessor = DataPreprocessor()
                    processed_data = preprocessor.prepare_formation_pressure_data(formation_data)
                    
                    # Split features and target
                    feature_cols = [col for col in processed_data.columns if col != 'FPress']
                    X = processed_data[feature_cols]
                    y = processed_data['FPress']
                    
                    # Initialize model
                    if model_type == "PCR":
                        model = PCRFormationPressure(n_components=n_components if show_advanced else 4)
                    elif model_type == "XGBoost":
                        model = XGBoostFormationPressure()
                    elif model_type == "Ensemble":
                        model = EnsembleFormationPressure(['pcr', 'xgboost'])
                    
                    # Train model
                    metrics = model.train(X, y, validation_split=validation_split if show_advanced else 0.2)